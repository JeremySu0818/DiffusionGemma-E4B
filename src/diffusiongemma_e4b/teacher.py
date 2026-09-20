from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import sqlite3
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from tqdm.auto import tqdm

from .constants import DEFAULT_LMSTUDIO_BASE_URL, DEFAULT_OLLAMA_BASE_URL
from .data_contract import TeacherSupervisedRecord, iter_jsonl
from .data_sources import iter_prompt_records


@dataclass
class TeacherConfig:
    runtime: str
    model: str
    base_url: str
    max_tokens: int | None
    temperature: float
    top_p: float
    timeout_s: int = 600
    max_retries: int = 5
    retry_base_s: float = 2.0
    min_estimated_tokens: int = 8
    tokenizer_name_or_path: str = ""
    tokenizer_revision: str = ""
    api_key: str = ""
    max_consecutive_failures: int = 20
    concurrency: int = 10
    prefetch_records: int = 16384
    prefetch_dir: Path | None = None
    student_prefix_length: int = 2048


class TeacherClient:
    def generate(self, prompt: str, media: dict[str, Any] | None = None) -> str:
        raise NotImplementedError


class _TeacherResponse(str):
    """Text plus optional runtime-native accounting returned by the server."""

    def __new__(
        cls,
        text: str,
        *,
        completion_tokens: int | None = None,
        request_seconds: float | None = None,
        generation_seconds: float | None = None,
    ) -> "_TeacherResponse":
        value = super().__new__(cls, text)
        value.completion_tokens = completion_tokens
        value.request_seconds = request_seconds
        value.generation_seconds = generation_seconds
        return value


@dataclass(frozen=True)
class _GenerationResult:
    text: str
    completion_tokens: int | None
    request_started: float
    request_finished: float
    generation_seconds: float | None = None

    @property
    def request_seconds(self) -> float:
        return max(0.0, self.request_finished - self.request_started)


class _ThroughputMeter:
    """Aggregate streamed token deltas on one shared monotonic timeline."""

    def __init__(self, window_seconds: float = 5.0, ewma_alpha: float = 0.2):
        self.window_seconds = window_seconds
        self.ewma_alpha = ewma_alpha
        self._bucket_tokens: dict[int, float] = {}
        self._request_tokens: dict[int, dict[int, float]] = {}
        self._next_request_id = 0
        self._lock = threading.Lock()

    def begin_request(self) -> int:
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._request_tokens[request_id] = {}
            return request_id

    def record_tokens(self, request_id: int, at: float, tokens: int) -> None:
        if tokens <= 0:
            return
        bucket = math.floor(at)
        with self._lock:
            request = self._request_tokens.get(request_id)
            if request is None:
                return
            request[bucket] = request.get(bucket, 0.0) + tokens
            self._bucket_tokens[bucket] = self._bucket_tokens.get(bucket, 0.0) + tokens
            self._prune_locked(at)

    def finish_request(self, request_id: int, completion_tokens: int | None) -> None:
        with self._lock:
            request = self._request_tokens.pop(request_id, {})
            estimated = sum(request.values())
            if completion_tokens is not None and completion_tokens >= 0 and estimated > 0:
                scale = completion_tokens / estimated
                for bucket, tokens in request.items():
                    self._bucket_tokens[bucket] = (
                        self._bucket_tokens.get(bucket, 0.0)
                        + tokens * (scale - 1.0)
                    )

    def cancel_request(self, request_id: int) -> None:
        # Generated tokens still consumed server throughput even when output
        # validation or the HTTP request later fails, so retain bucket counts.
        with self._lock:
            self._request_tokens.pop(request_id, None)

    def _prune_locked(self, now: float) -> None:
        oldest = math.floor(now - self.window_seconds - 2.0)
        for bucket in tuple(self._bucket_tokens):
            if bucket < oldest:
                del self._bucket_tokens[bucket]

    def total_tps(self, now: float | None = None) -> float | None:
        now = time.monotonic() if now is None else now
        last_complete = math.floor(now) - 1
        first = last_complete - math.ceil(self.window_seconds) + 1
        with self._lock:
            self._prune_locked(now)
            values = [self._bucket_tokens.get(bucket, 0.0) for bucket in range(first, last_complete + 1)]
        if not values or not any(values):
            return None
        first_nonzero = next(index for index, value in enumerate(values) if value > 0)
        smoothed = values[first_nonzero]
        for value in values[first_nonzero + 1:]:
            smoothed = self.ewma_alpha * value + (1.0 - self.ewma_alpha) * smoothed
        return smoothed


def _throughput_eta(
    target_tokens: int,
    current_tokens: int,
    total_tps: float | None,
) -> str:
    if total_tps is None or total_tps <= 0:
        return "?"
    remaining_tokens = max(0, target_tokens - current_tokens)
    return tqdm.format_interval(remaining_tokens / total_tps)


def _response_text(data: Any) -> str:
    if not isinstance(data, dict):
        raise ValueError("teacher response is not a JSON object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("teacher response has no choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("teacher response has no choices[0].message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
        return "".join(parts).strip()
    raise ValueError("teacher response message content is not text")


class OpenAICompletionsClient(TeacherClient):
    """Chat completions against vLLM, LM Studio, or llama.cpp compatible APIs."""

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.session = requests.Session()
        if cfg.api_key:
            self.session.headers.update({"Authorization": f"Bearer {cfg.api_key}"})

    def generate(
        self,
        prompt: str,
        media: dict[str, Any] | None = None,
        on_delta: Callable[[str, float], None] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": _chat_content(prompt, media or {})}],
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.cfg.max_tokens is not None:
            payload["max_tokens"] = self.cfg.max_tokens
        started = time.monotonic()
        response = self.session.post(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=self.cfg.timeout_s,
            stream=True,
        )
        response.raise_for_status()
        parts: list[str] = []
        completion_tokens: int | None = None
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            line = str(line or "").strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if payload_text == "[DONE]":
                break
            event = json.loads(payload_text)
            usage = event.get("usage") if isinstance(event, dict) else None
            reported = usage.get("completion_tokens") if isinstance(usage, dict) else None
            if isinstance(reported, int) and reported >= 0:
                completion_tokens = reported
            choices = event.get("choices") if isinstance(event, dict) else None
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            reasoning = (
                delta.get("reasoning_content") if isinstance(delta, dict) else None
            )
            if isinstance(reasoning, str) and reasoning and on_delta is not None:
                on_delta(reasoning, time.monotonic())
            if isinstance(content, str) and content:
                parts.append(content)
                if on_delta is not None:
                    on_delta(content, time.monotonic())
        request_seconds = time.monotonic() - started
        if not isinstance(completion_tokens, int) or completion_tokens < 0:
            completion_tokens = None
        return _TeacherResponse(
            "".join(parts).strip(),
            completion_tokens=completion_tokens,
            request_seconds=request_seconds,
        )


class OllamaGenerateClient(TeacherClient):
    """Ollama chat API client (text-only in this pipeline)."""

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def generate(self, prompt: str, media: dict[str, Any] | None = None) -> str:
        if media:
            raise RuntimeError("Ollama multimodal generation is not supported by this pipeline; use the vLLM endpoint.")
        options: dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        if self.cfg.max_tokens is not None:
            options["num_predict"] = self.cfg.max_tokens
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }
        started = time.monotonic()
        response = self.session.post(
            f"{self.cfg.base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=self.cfg.timeout_s,
        )
        request_seconds = time.monotonic() - started
        response.raise_for_status()
        data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Ollama teacher response has no message.content")
        completion_tokens = data.get("eval_count")
        if not isinstance(completion_tokens, int) or completion_tokens < 0:
            completion_tokens = None
        eval_duration = data.get("eval_duration")
        generation_seconds = (
            float(eval_duration) / 1_000_000_000
            if isinstance(eval_duration, (int, float)) and eval_duration > 0
            else None
        )
        return _TeacherResponse(
            message["content"].strip(),
            completion_tokens=completion_tokens,
            request_seconds=request_seconds,
            generation_seconds=generation_seconds,
        )


def make_client(cfg: TeacherConfig) -> TeacherClient:
    if cfg.runtime in {"lmstudio", "llamacpp", "openai-compatible"}:
        return OpenAICompletionsClient(cfg)
    if cfg.runtime == "ollama":
        return OllamaGenerateClient(cfg)
    raise ValueError(f"unsupported runtime: {cfg.runtime}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _media_url(value: Any, prefer_data_url: bool = False) -> str:
    if isinstance(value, dict):
        value = value.get("path") or value.get("url")
    value = str(value)
    if value.startswith(("http://", "https://", "data:", "file://")):
        return value
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"teacher media file does not exist: {path}")
    if prefer_data_url:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
    return path.as_uri()


def _chat_content(prompt: str, media: dict[str, Any]) -> str | list[dict[str, Any]]:
    if not media:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for value in _as_list(media.get("image")) + _as_list(media.get("images")):
        content.append({"type": "image_url", "image_url": {"url": _media_url(value, prefer_data_url=True)}})
    for value in _as_list(media.get("audio")) + _as_list(media.get("audios")):
        content.append({"type": "audio_url", "audio_url": {"url": _media_url(value)}})
    return content


def estimate_tokens(text: str) -> int:
    # Formal accounting is performed with the student tokenizer during corruption.
    return max(1, len(text.encode("utf-8")) // 4)


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_fingerprint(cfg: TeacherConfig, data_fingerprint: str = "") -> str:
    return _sha256_json(
        {
            "schema": 2,
            "runtime": cfg.runtime,
            "model": cfg.model,
            "base_url": cfg.base_url.rstrip("/"),
            # ``None`` means no API output cap.  Keep the legacy default in
            # the resume identity so existing 4096-cap datasets can continue
            # without splitting a durable output file into two fingerprints.
            "max_tokens": cfg.max_tokens if cfg.max_tokens is not None else 4096,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "min_estimated_tokens": cfg.min_estimated_tokens,
            "tokenizer": cfg.tokenizer_name_or_path,
            "tokenizer_revision": cfg.tokenizer_revision,
            "student_prefix_length": cfg.student_prefix_length,
            "data_fingerprint": data_fingerprint,
        }
    )


def _repair_jsonl_tail(path: Path) -> None:
    """Keep complete final JSON and truncate only a crash-partial final line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r+b") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
        last_newline = data.rfind(b"\n")
        fragment = data[last_newline + 1 :]
        try:
            json.loads(fragment.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            f.truncate(last_newline + 1 if last_newline >= 0 else 0)
        else:
            f.seek(0, os.SEEK_END)
            f.write(b"\n")
        f.flush()
        os.fsync(f.fileno())


def progress_from_output(output_path: Path) -> dict[str, Any]:
    if not output_path.exists():
        return {"records": 0, "estimated_tokens": 0, "source_index": 0}
    _repair_jsonl_tail(output_path)
    records = 0
    estimated_tokens = 0
    source_index = 0
    last_record_id = None
    fingerprints: set[str] = set()
    records_by_bucket: dict[str, int] = {}
    tokens_by_bucket: dict[str, int] = {}
    for row in iter_jsonl(output_path):
        records += 1
        estimated_tokens += int(row.get("estimated_tokens") or 0)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        source_index = max(source_index, int(metadata.get("source_index") or records))
        fingerprint = str(metadata.get("generation_fingerprint") or "")
        bucket = str(metadata.get("bucket") or "unknown")
        records_by_bucket[bucket] = records_by_bucket.get(bucket, 0) + 1
        tokens_by_bucket[bucket] = tokens_by_bucket.get(bucket, 0) + int(
            row.get("estimated_tokens") or 0
        )
        if fingerprint:
            fingerprints.add(fingerprint)
        last_record_id = row.get("id") or last_record_id
    state: dict[str, Any] = {
        "records": records,
        "estimated_tokens": estimated_tokens,
        "source_index": source_index,
        "records_by_bucket": records_by_bucket,
        "estimated_tokens_by_bucket": tokens_by_bucket,
    }
    if last_record_id is not None:
        state["last_record_id"] = last_record_id
    if len(fingerprints) == 1:
        state["generation_fingerprint"] = next(iter(fingerprints))
    elif len(fingerprints) > 1:
        raise RuntimeError(f"teacher output contains multiple generation fingerprints: {sorted(fingerprints)}")
    return state


def read_progress(
    progress_path: Path,
    output_path: Path | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    output_state = progress_from_output(output_path) if output_path is not None else {"records": 0, "estimated_tokens": 0}
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        state = {"records": 0, "estimated_tokens": 0, "source_index": 0}
    if expected_fingerprint and (int(state.get("records", 0)) > 0 or int(output_state.get("records", 0)) > 0):
        found = output_state.get("generation_fingerprint") or state.get("generation_fingerprint")
        if found != expected_fingerprint:
            raise RuntimeError(
                "teacher resume fingerprint mismatch; use a new output directory or restore the original data/model/config"
            )
    if (
        output_state["records"] != int(state.get("records", 0))
        or output_state["estimated_tokens"] != int(state.get("estimated_tokens", 0))
        or int(output_state.get("source_index", 0)) > int(state.get("source_index", 0))
    ):
        state.update(output_state)
        state["reconciled_from_output"] = True
    elif int(output_state.get("records", 0)) > 0:
        # Schema additions and crash reconciliation must also restore aggregate
        # mix counters even when the scalar progress totals already match.
        state["records_by_bucket"] = output_state.get("records_by_bucket", {})
        state["estimated_tokens_by_bucket"] = output_state.get(
            "estimated_tokens_by_bucket", {}
        )
    if expected_fingerprint:
        state["generation_fingerprint"] = expected_fingerprint
    return state


def save_progress(progress_path: Path, state: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{progress_path.name}.", suffix=".tmp", dir=progress_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, progress_path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _build_prompt(record: dict[str, Any]) -> str:
    prompt = str(record.get("prompt_text") or record.get("prompt") or "").strip()
    context = str(record.get("context_text") or record.get("context") or "").strip()
    if context and prompt:
        return f"Context:\n{context}\n\nRequest:\n{prompt}"
    if context:
        return context
    if prompt:
        return prompt
    raise ValueError("prompt record must contain prompt_text/prompt or context_text/context")


_ENDPOINT_FAILURE_RE = re.compile(
    r"(?:internal server error|bad gateway|service unavailable|upstream error|rate limit exceeded|<html)", re.IGNORECASE
)


def _validated_teacher_text(
    text: str,
    prompt: str,
    min_estimated_tokens: int,
    token_counter: Callable[[str], int] = estimate_tokens,
) -> str:
    text = text.strip()
    if not text:
        raise ValueError("teacher returned empty text")
    if token_counter(text) < min_estimated_tokens:
        raise ValueError(f"teacher output is shorter than {min_estimated_tokens} estimated tokens")
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    prompt_normalized = re.sub(r"\s+", " ", prompt).strip().casefold()
    if normalized == prompt_normalized:
        raise ValueError("teacher echoed the prompt without an answer")
    if _ENDPOINT_FAILURE_RE.search(text):
        raise ValueError("teacher output contains endpoint failure text")
    words = normalized.split()
    if len(words) >= 32 and len(set(words)) / len(words) < 0.08:
        raise ValueError("teacher output is pathologically repetitive")
    return text


def _generate_with_retry(
    client: TeacherClient,
    cfg: TeacherConfig,
    prompt: str,
    media: dict[str, Any],
    token_counter: Callable[[str], int] = estimate_tokens,
    throughput: _ThroughputMeter | None = None,
) -> _GenerationResult:
    errors: list[str] = []
    for attempt in range(cfg.max_retries + 1):
        request_id: int | None = None
        try:
            request_started = time.monotonic()
            if throughput is not None and isinstance(client, OpenAICompletionsClient):
                request_id = throughput.begin_request()

                def on_delta(content: str, at: float) -> None:
                    throughput.record_tokens(
                        request_id,
                        at,
                        max(1, token_counter(content)),
                    )

                raw = client.generate(prompt, media=media, on_delta=on_delta)
                throughput.finish_request(
                    request_id,
                    getattr(raw, "completion_tokens", None),
                )
                request_id = None
            else:
                raw = client.generate(prompt, media=media)
            request_finished = time.monotonic()
            text = _validated_teacher_text(
                str(raw), prompt, cfg.min_estimated_tokens, token_counter
            )
            reported_seconds = getattr(raw, "request_seconds", None)
            if isinstance(reported_seconds, (int, float)) and reported_seconds >= 0:
                request_started = request_finished - float(reported_seconds)
            return _GenerationResult(
                text=text,
                completion_tokens=getattr(raw, "completion_tokens", None),
                request_started=request_started,
                request_finished=request_finished,
                generation_seconds=getattr(raw, "generation_seconds", None),
            )
        except (requests.RequestException, ValueError, RuntimeError, OSError) as exc:
            if throughput is not None and request_id is not None:
                throughput.cancel_request(request_id)
            errors.append(f"attempt={attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt >= cfg.max_retries:
                break
            base_delay = min(60.0, cfg.retry_base_s * (2**attempt))
            delay = min(60.0, random.uniform(0.5, 1.5) * base_delay) if base_delay > 0 else 0.0
            time.sleep(delay)
    raise RuntimeError("teacher generation failed after retries: " + " | ".join(errors))


def _existing_output_sets(output_path: Path | None) -> tuple[set[str], set[str]]:
    prompt_ids: set[str] = set()
    text_hashes: set[str] = set()
    if output_path is None or not output_path.exists():
        return prompt_ids, text_hashes
    _repair_jsonl_tail(output_path)
    for row in iter_jsonl(output_path):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        prompt_id = str(metadata.get("prompt_record_id") or "")
        if prompt_id:
            prompt_ids.add(prompt_id)
        text_hashes.add(hashlib.sha256(re.sub(r"\s+", " ", str(row.get("text") or "")).strip().encode("utf-8")).hexdigest())
    return prompt_ids, text_hashes


def _record_bucket(prompt_record: dict[str, Any]) -> str:
    metadata = (
        prompt_record.get("metadata")
        if isinstance(prompt_record.get("metadata"), dict)
        else {}
    )
    return str(prompt_record.get("bucket") or metadata.get("bucket") or "unknown")


def _identity_prompt_limiter(prompt: str, _media: dict[str, Any]) -> str:
    return prompt


def _commit_bucket_progress(
    state: dict[str, Any],
    prompt_record: dict[str, Any],
    token_count: int,
) -> None:
    bucket = _record_bucket(prompt_record)
    records = dict(state.get("records_by_bucket") or {})
    tokens = dict(state.get("estimated_tokens_by_bucket") or {})
    records[bucket] = int(records.get(bucket, 0)) + 1
    tokens[bucket] = int(tokens.get(bucket, 0)) + int(token_count)
    state["records_by_bucket"] = records
    state["estimated_tokens_by_bucket"] = tokens


def validate_generated_mix(
    state: dict[str, Any],
    source_config: dict[str, Any],
) -> dict[str, Any]:
    """Fail if a required prompt bucket silently disappears at the teacher API."""

    expected = {
        str(item["bucket"]): float(item["share"])
        for item in source_config.get("recommended_mix", [])
        if item.get("required", True) and float(item.get("share") or 0.0) > 0
    }
    actual_counts = {
        str(key): int(value)
        for key, value in dict(state.get("records_by_bucket") or {}).items()
    }
    total = sum(actual_counts.values())
    if total <= 0:
        raise RuntimeError("teacher generation produced no durable records")
    missing = sorted(bucket for bucket in expected if actual_counts.get(bucket, 0) <= 0)
    if missing:
        raise RuntimeError(
            "teacher generation lost required dataset bucket(s): "
            + ", ".join(missing)
        )
    actual_shares = {
        bucket: actual_counts.get(bucket, 0) / total for bucket in expected
    }
    severely_underfilled = {
        bucket: {
            "expected_prompt_share": share,
            "actual_prompt_share": actual_shares[bucket],
        }
        for bucket, share in expected.items()
        if actual_shares[bucket] + (1.0 / total) < share * 0.5
    }
    if severely_underfilled:
        raise RuntimeError(
            "teacher success rates distorted the required prompt mixture by more "
            f"than 50%: {severely_underfilled}"
        )
    return {
        "expected_prompt_shares": expected,
        "actual_prompt_shares": actual_shares,
        "records_by_bucket": actual_counts,
        "estimated_tokens_by_bucket": dict(
            state.get("estimated_tokens_by_bucket") or {}
        ),
    }


def _generate_records_sync(
    cfg: TeacherConfig,
    prompt_records: Iterable[dict[str, Any]],
    target_estimated_tokens: int,
    progress_path: Path,
    output_path: Path | None = None,
    data_fingerprint: str = "",
    token_counter: Callable[[str], int] = estimate_tokens,
    prompt_limiter: Callable[[str, dict[str, Any]], str] = _identity_prompt_limiter,
    throughput: _ThroughputMeter | None = None,
) -> Iterable[TeacherSupervisedRecord]:
    client = make_client(cfg)
    fingerprint = generation_fingerprint(cfg, data_fingerprint)
    state = read_progress(progress_path, output_path=output_path, expected_fingerprint=fingerprint)
    state.setdefault("records", 0)
    state.setdefault("estimated_tokens", 0)
    state.setdefault("source_index", 0)
    state.setdefault("filtered_records", 0)
    save_progress(progress_path, state)
    existing_prompt_ids, seen_text_hashes = _existing_output_sets(output_path)
    consecutive_failures = int(state.get("consecutive_failures", 0))

    for source_index, prompt_record in enumerate(prompt_records):
        if target_estimated_tokens > 0 and int(state["estimated_tokens"]) >= target_estimated_tokens:
            break
        prompt_record_id = str(prompt_record.get("id") or _sha256_json(prompt_record))
        # Resume is keyed by stable prompt IDs. A positional source_index is only
        # audit metadata because streaming source failures can change positions.
        if prompt_record_id in existing_prompt_ids:
            continue
        started = time.time()
        prompt = _build_prompt(prompt_record)
        media = dict(prompt_record.get("media") or {})
        prompt = prompt_limiter(prompt, media)
        try:
            generation = _generate_with_retry(
                client, cfg, prompt, media, token_counter, throughput
            )
            text = generation.text
        except RuntimeError as exc:
            consecutive_failures += 1
            state["failed_prompts"] = int(state.get("failed_prompts", 0)) + 1
            state["consecutive_failures"] = consecutive_failures
            state["source_index"] = source_index + 1
            state["last_failed_prompt_id"] = prompt_record_id
            state["last_failure"] = str(exc)
            save_progress(progress_path, state)
            if consecutive_failures >= cfg.max_consecutive_failures:
                raise RuntimeError(
                    f"teacher failed {consecutive_failures} consecutive prompts; aborting to avoid silent underfill"
                ) from exc
            if cfg.retry_base_s > 0:
                cooldown = min(
                    30.0,
                    random.uniform(0.5, 1.5) * cfg.retry_base_s * min(consecutive_failures, 5),
                )
                time.sleep(cooldown)
            continue
        consecutive_failures = 0
        state["consecutive_failures"] = 0
        tok = token_counter(text)
        text_hash = hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            state["filtered_records"] = int(state.get("filtered_records", 0)) + 1
            state["source_index"] = source_index + 1
            state["last_filter_reason"] = "duplicate_teacher_output"
            save_progress(progress_path, state)
            continue

        metadata = {str(k): v for k, v in dict(prompt_record.get("metadata") or {}).items()}
        metadata.update(
            {
                "prompt_record_id": prompt_record_id,
                "source_index": source_index + 1,
                "source_name": prompt_record.get("source_name"),
                "bucket": prompt_record.get("bucket") or metadata.get("bucket"),
                "generation_fingerprint": fingerprint,
                "teacher_temperature": cfg.temperature,
                "teacher_top_p": cfg.top_p,
                "teacher_max_tokens": cfg.max_tokens,
                "teacher_input_tokens": token_counter(prompt),
                "teacher_completion_tokens": generation.completion_tokens,
                "teacher_request_seconds": round(generation.request_seconds, 6),
                "teacher_generation_seconds": (
                    round(generation.generation_seconds, 6)
                    if generation.generation_seconds is not None
                    else None
                ),
            }
        )
        record_id = "teacher-" + _sha256_json(
            {"prompt_record_id": prompt_record_id, "generation_fingerprint": fingerprint}
        )
        record = TeacherSupervisedRecord(
            id=record_id,
            source_model=cfg.model,
            runtime=cfg.runtime,
            prompt_text=str(
                prompt_record.get("prompt_text")
                or prompt_record.get("prompt")
                or ""
            ),
            text=text,
            estimated_tokens=tok,
            prompt_source=str(prompt_record.get("source") or prompt_record.get("prompt_source") or "dataset_prompt_bank"),
            modality=str(prompt_record.get("modality") or "text"),
            context_text=str(prompt_record.get("context_text") or prompt_record.get("context") or ""),
            media={str(k): v for k, v in media.items()},
            metadata=metadata,
        )
        # The caller appends and fsyncs this record before requesting the next
        # generator item. State is committed only after that durable append.
        yield record
        seen_text_hashes.add(text_hash)
        existing_prompt_ids.add(prompt_record_id)
        state["records"] = int(state["records"]) + 1
        state["estimated_tokens"] = int(state["estimated_tokens"]) + tok
        state["source_index"] = source_index + 1
        state["last_record_id"] = record.id
        state["last_seconds"] = round(time.time() - started, 3)
        state["last_estimated_tokens"] = tok
        _commit_bucket_progress(state, prompt_record, tok)
        save_progress(progress_path, state)

    if target_estimated_tokens > 0 and int(state["estimated_tokens"]) < target_estimated_tokens:
        raise RuntimeError(
            f"teacher dataset underfilled: have {state['estimated_tokens']} estimated tokens, "
            f"need {target_estimated_tokens}; inspect the dataset source failure summary"
        )


@dataclass
class _PendingGeneration:
    source_index: int
    prompt_record: dict[str, Any]
    prompt_record_id: str
    prompt: str
    media: dict[str, Any]
    started: float
    future: Future[_GenerationResult]


_worker_local = threading.local()

# A streaming dataset can take a little while to open the next shard, but it
# must never leave the generation scheduler waiting forever with LM Studio
# idle.  Keep this deliberately finite so the progress file and terminal say
# which side of the pipeline is stalled.
_PROMPT_PREFETCH_STALL_TIMEOUT_S = 900.0
# Wake periodically while requests are running so newly spooled prompts can
# occupy idle teacher slots.  Without this timeout, startup could block on the
# first request even after the producer had filled the disk spool.
_PROMPT_SLOT_REFILL_INTERVAL_S = 0.05


class _DiskPromptQueue:
    """A bounded, resumable cross-thread queue whose payloads live on disk."""

    def __init__(self, directory: Path, max_records: int, fingerprint: str = "default"):
        directory.mkdir(parents=True, exist_ok=True)
        safe_fingerprint = re.sub(r"[^a-zA-Z0-9_.-]", "_", fingerprint)[:64]
        self.path = directory / f"prompts-{safe_fingerprint}.sqlite3"
        self._condition = threading.Condition()
        self._closed = False
        self._error: BaseException | None = None
        self._max_records = max_records
        self._writer = sqlite3.connect(self.path, check_same_thread=False)
        self._reader = sqlite3.connect(self.path, check_same_thread=False)
        # WAL lets the producer append while the scheduler dequeues. NORMAL
        # synchronous mode avoids a full disk flush for every prefetched prompt;
        # teacher_outputs.jsonl remains the per-record durable source of truth.
        self._writer.execute("PRAGMA journal_mode=WAL")
        for connection in (self._writer, self._reader):
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA temp_store=MEMORY")
        columns = {
            str(row[1])
            for row in self._writer.execute("PRAGMA table_info(prompts)").fetchall()
        }
        if columns and "prompt_key" not in columns:
            # Migrate position-keyed spools. Parallel source prefetch can alter
            # arrival positions after a restart, while prompt IDs remain stable.
            self._writer.execute("DROP INDEX IF EXISTS prompts_status_source_idx")
            self._writer.execute("ALTER TABLE prompts RENAME TO prompts_legacy")
            self._writer.execute(
                "CREATE TABLE prompts ("
                "prompt_key TEXT PRIMARY KEY, source_index INTEGER NOT NULL, "
                "payload TEXT NOT NULL, "
                "status TEXT NOT NULL CHECK(status IN ('queued', 'inflight', 'consumed')))"
            )
            legacy_rows = self._writer.execute(
                "SELECT source_index, payload, status FROM prompts_legacy"
            )
            for source_index, payload, status in legacy_rows:
                prompt_record = json.loads(payload)
                prompt_key = str(prompt_record.get("id") or _sha256_json(prompt_record))
                self._writer.execute(
                    "INSERT OR IGNORE INTO prompts(prompt_key, source_index, payload, status) "
                    "VALUES (?, ?, ?, ?)",
                    (prompt_key, source_index, payload, status),
                )
            self._writer.execute("DROP TABLE prompts_legacy")
        else:
            self._writer.execute(
                "CREATE TABLE IF NOT EXISTS prompts ("
                "prompt_key TEXT PRIMARY KEY, source_index INTEGER NOT NULL, "
                "payload TEXT NOT NULL, "
                "status TEXT NOT NULL CHECK(status IN ('queued', 'inflight', 'consumed')))"
            )
        # Older versions retained every acknowledged row as ``consumed``,
        # causing unbounded database growth. Completed prompt identity already
        # lives durably in teacher_outputs.jsonl, so the spool must contain
        # queued/inflight work only.
        self._writer.execute("DELETE FROM prompts WHERE status = 'consumed'")
        # A process may have stopped after dequeueing but before durably writing
        # its teacher output. Make those rows available again on restart.
        self._writer.execute("UPDATE prompts SET status = 'queued' WHERE status = 'inflight'")
        self._writer.execute(
            "CREATE INDEX IF NOT EXISTS prompts_status_source_idx "
            "ON prompts(status, source_index)"
        )
        self._writer.commit()
        row = self._writer.execute("SELECT COUNT(*) FROM prompts").fetchone()
        self._active_records = int(row[0]) if row else 0

    def _active_count(self) -> int:
        return self._active_records

    def put(self, source_index: int, prompt_record: dict[str, Any], stop: threading.Event) -> bool:
        payload = json.dumps(prompt_record, ensure_ascii=False, separators=(",", ":"))
        prompt_key = str(prompt_record.get("id") or _sha256_json(prompt_record))
        with self._condition:
            while not stop.is_set() and self._active_count() >= self._max_records:
                self._condition.wait(timeout=0.1)
            if stop.is_set():
                return False
            cursor = self._writer.execute(
                "INSERT OR IGNORE INTO prompts(prompt_key, source_index, payload, status) "
                "VALUES (?, ?, ?, 'queued')",
                (prompt_key, source_index, payload),
            )
            self._writer.commit()
            self._active_records += max(0, cursor.rowcount)
            self._condition.notify_all()
            return True

    def finish(self, error: BaseException | None = None) -> None:
        with self._condition:
            self._error = error
            self._closed = True
            self._condition.notify_all()

    def get(self, timeout: float = 0.0) -> tuple[int, dict[str, Any]] | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            row = None
            while row is None and not self._closed:
                row = self._reader.execute(
                    "SELECT prompt_key, source_index, payload FROM prompts "
                    "WHERE status = 'queued' ORDER BY source_index, prompt_key LIMIT 1"
                ).fetchone()
                if row is not None:
                    break
                remaining = deadline - time.monotonic()
                if timeout <= 0 or remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            if row is None:
                row = self._reader.execute(
                    "SELECT prompt_key, source_index, payload FROM prompts "
                    "WHERE status = 'queued' ORDER BY source_index, prompt_key LIMIT 1"
                ).fetchone()
            if row is None:
                if self._error is not None:
                    raise self._error
                return None
            self._reader.execute(
                "UPDATE prompts SET status = 'inflight' WHERE prompt_key = ?", (row[0],)
            )
            self._reader.commit()
        return int(row[1]), json.loads(row[2])

    def acknowledge(self, prompt_key: str) -> None:
        with self._condition:
            cursor = self._reader.execute(
                "DELETE FROM prompts WHERE prompt_key = ?", (prompt_key,)
            )
            self._reader.commit()
            self._active_records = max(0, self._active_records - max(0, cursor.rowcount))
            self._condition.notify_all()

    def requeue_inflight(self) -> None:
        with self._condition:
            self._reader.execute("UPDATE prompts SET status = 'queued' WHERE status = 'inflight'")
            self._reader.commit()
            self._condition.notify_all()

    @property
    def exhausted(self) -> bool:
        with self._condition:
            row = self._reader.execute(
                "SELECT 1 FROM prompts WHERE status = 'queued' LIMIT 1"
            ).fetchone()
            return self._closed and row is None

    def close(self) -> None:
        self._writer.close()
        self._reader.close()


def _worker_generate(
    cfg: TeacherConfig,
    prompt: str,
    media: dict[str, Any],
    token_counter: Callable[[str], int],
    throughput: _ThroughputMeter | None = None,
) -> _GenerationResult:
    """Generate in a worker-local client so requests.Session is never shared."""
    client = getattr(_worker_local, "client", None)
    client_cfg_id = getattr(_worker_local, "client_cfg_id", None)
    if client is None or client_cfg_id != id(cfg):
        client = make_client(cfg)
        _worker_local.client = client
        _worker_local.client_cfg_id = id(cfg)
    return _generate_with_retry(
        client, cfg, prompt, media, token_counter, throughput
    )


def _generate_records_concurrent(
    cfg: TeacherConfig,
    prompt_records: Iterable[dict[str, Any]],
    target_estimated_tokens: int,
    progress_path: Path,
    output_path: Path | None = None,
    data_fingerprint: str = "",
    token_counter: Callable[[str], int] = estimate_tokens,
    prompt_limiter: Callable[[str, dict[str, Any]], str] = _identity_prompt_limiter,
    throughput: _ThroughputMeter | None = None,
) -> Iterable[TeacherSupervisedRecord]:
    fingerprint = generation_fingerprint(cfg, data_fingerprint)
    state = read_progress(progress_path, output_path=output_path, expected_fingerprint=fingerprint)
    state.setdefault("records", 0)
    state.setdefault("estimated_tokens", 0)
    state.setdefault("source_index", 0)
    state.setdefault("filtered_records", 0)
    save_progress(progress_path, state)
    existing_prompt_ids, seen_text_hashes = _existing_output_sets(output_path)
    scheduled_prompt_ids = set(existing_prompt_ids)
    consecutive_failures = int(state.get("consecutive_failures", 0))
    # Dataset streaming and media preparation may block on disk or the network.
    # Keep that work off the scheduler thread so a completed teacher request can
    # be replaced immediately from this bounded, single-producer disk spool.
    # Keep a bounded on-disk reservoir so transient streaming latency does
    # not leave local inference slots idle.  This is intentionally independent
    # of concurrency: four workers need more than eight ready prompts when a
    # remote shard pauses between records.
    prefetch_size = max(cfg.prefetch_records, cfg.concurrency + 1)
    spool_root = cfg.prefetch_dir or progress_path.parent / "prompt_spool"
    source_queue = _DiskPromptQueue(
        Path(spool_root), max_records=prefetch_size, fingerprint=fingerprint
    )
    source_stop = threading.Event()

    def prefetch_prompt_records() -> None:
        # ``datasets`` uses tqdm while opening streaming shards.  That bar runs
        # concurrently with the teacher progress bar and otherwise overwrites
        # it in the terminal, making generation appear to have no progress.
        try:
            from datasets.utils.logging import disable_progress_bar

            disable_progress_bar()
        except ImportError:
            pass
        source_items = enumerate(iter(prompt_records))
        error: BaseException | None = None
        try:
            for source_index, prompt_record in source_items:
                if not source_queue.put(source_index, prompt_record, source_stop):
                    return
        except BaseException as exc:  # Propagate source failures on the scheduler thread.
            error = exc
        finally:
            close = getattr(source_items, "close", None)
            if callable(close):
                close()
            source_queue.finish(error)

    producer = threading.Thread(
        target=prefetch_prompt_records,
        name="teacher-prompt-prefetch",
        daemon=True,
    )
    producer.start()
    source_exhausted = False
    reached_target = False
    last_source_activity = time.monotonic()
    pending: deque[_PendingGeneration] = deque()
    executor = ThreadPoolExecutor(max_workers=cfg.concurrency, thread_name_prefix="teacher-request")

    def next_source_item(wait_s: float = 0.0) -> tuple[int, dict[str, Any]] | None:
        nonlocal last_source_activity, source_exhausted
        value = source_queue.get(timeout=wait_s)
        if value is None:
            source_exhausted = source_queue.exhausted
            return None
        # Resume may consume many already-durable prompt IDs before reaching
        # the first new one.  Those items prove the source is making progress
        # and must reset the stall clock even though they are not scheduled.
        last_source_activity = time.monotonic()
        return value

    def fill_window(wait_s: float = 0.0) -> None:
        nonlocal source_exhausted
        while not source_exhausted and len(pending) < cfg.concurrency:
            if target_estimated_tokens > 0 and int(state["estimated_tokens"]) >= target_estimated_tokens:
                return
            # Once a request is in flight, never wait for a slow source here:
            # doing so used to prevent completed requests from being appended.
            source_item = next_source_item(wait_s=wait_s if not pending else 0.0)
            if source_item is None:
                return
            source_index, prompt_record = source_item
            prompt_record_id = str(prompt_record.get("id") or _sha256_json(prompt_record))
            if prompt_record_id in scheduled_prompt_ids:
                source_queue.acknowledge(prompt_record_id)
                continue
            prompt = _build_prompt(prompt_record)
            media = dict(prompt_record.get("media") or {})
            prompt = prompt_limiter(prompt, media)
            scheduled_prompt_ids.add(prompt_record_id)
            pending.append(
                _PendingGeneration(
                    source_index=source_index,
                    prompt_record=prompt_record,
                    prompt_record_id=prompt_record_id,
                    prompt=prompt,
                    media=media,
                    started=time.time(),
                    future=executor.submit(
                        _worker_generate,
                        cfg,
                        prompt,
                        media,
                        token_counter,
                        throughput,
                    ),
                )
            )

    try:
        while pending or not source_exhausted:
            fill_window()
            if not pending:
                if source_exhausted:
                    break
                if time.monotonic() - last_source_activity >= _PROMPT_PREFETCH_STALL_TIMEOUT_S:
                    raise RuntimeError(
                        "prompt source prefetch stalled for "
                        f"{_PROMPT_PREFETCH_STALL_TIMEOUT_S:g}s; no new prompt was available"
                    )
                # There is no LM request to service, so a short bounded wait
                # is appropriate.  It is repeated only until the explicit
                # stall timeout above, rather than forever.
                fill_window(wait_s=0.1)
                continue

            # Commit in completion order. Waiting only on pending[0] caused
            # head-of-line blocking: faster responses accumulated in RAM,
            # no slots were refilled, and four-way LM Studio concurrency
            # eventually collapsed to one long request.
            completed, _ = wait(
                [candidate.future for candidate in pending],
                timeout=_PROMPT_SLOT_REFILL_INTERVAL_S,
                return_when=FIRST_COMPLETED,
            )
            if not completed:
                fill_window()
                continue
            item = min(
                (candidate for candidate in pending if candidate.future in completed),
                key=lambda candidate: candidate.source_index,
            )
            try:
                generation = item.future.result()
                text = generation.text
            except RuntimeError as exc:
                pending.remove(item)
                consecutive_failures += 1
                state["failed_prompts"] = int(state.get("failed_prompts", 0)) + 1
                state["consecutive_failures"] = consecutive_failures
                state["source_index"] = max(
                    int(state.get("source_index", 0)), item.source_index + 1
                )
                state["last_failed_prompt_id"] = item.prompt_record_id
                state["last_failure"] = str(exc)
                save_progress(progress_path, state)
                source_queue.acknowledge(item.prompt_record_id)
                if consecutive_failures >= cfg.max_consecutive_failures:
                    raise RuntimeError(
                        f"teacher failed {consecutive_failures} consecutive prompts; aborting to avoid silent underfill"
                    ) from exc
                if cfg.retry_base_s > 0:
                    cooldown = min(
                        30.0,
                        random.uniform(0.5, 1.5) * cfg.retry_base_s * min(consecutive_failures, 5),
                    )
                    time.sleep(cooldown)
                fill_window()
                continue

            consecutive_failures = 0
            state["consecutive_failures"] = 0
            tok = token_counter(text)
            text_hash = hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()
            if text_hash in seen_text_hashes:
                pending.remove(item)
                state["filtered_records"] = int(state.get("filtered_records", 0)) + 1
                state["source_index"] = max(
                    int(state.get("source_index", 0)), item.source_index + 1
                )
                state["last_filter_reason"] = "duplicate_teacher_output"
                save_progress(progress_path, state)
                source_queue.acknowledge(item.prompt_record_id)
                fill_window()
                continue

            metadata = {str(k): v for k, v in dict(item.prompt_record.get("metadata") or {}).items()}
            metadata.update(
                {
                    "prompt_record_id": item.prompt_record_id,
                    "source_index": item.source_index + 1,
                    "source_name": item.prompt_record.get("source_name"),
                    "bucket": item.prompt_record.get("bucket") or metadata.get("bucket"),
                    "generation_fingerprint": fingerprint,
                    "teacher_temperature": cfg.temperature,
                    "teacher_top_p": cfg.top_p,
                    "teacher_max_tokens": cfg.max_tokens,
                    "teacher_input_tokens": token_counter(item.prompt),
                    "teacher_completion_tokens": generation.completion_tokens,
                    "teacher_request_seconds": round(generation.request_seconds, 6),
                    "teacher_generation_seconds": (
                        round(generation.generation_seconds, 6)
                        if generation.generation_seconds is not None
                        else None
                    ),
                }
            )
            record = TeacherSupervisedRecord(
                id="teacher-"
                + _sha256_json(
                    {"prompt_record_id": item.prompt_record_id, "generation_fingerprint": fingerprint}
                ),
                source_model=cfg.model,
                runtime=cfg.runtime,
                prompt_text=str(
                    item.prompt_record.get("prompt_text")
                    or item.prompt_record.get("prompt")
                    or ""
                ),
                text=text,
                estimated_tokens=tok,
                prompt_source=str(
                    item.prompt_record.get("source")
                    or item.prompt_record.get("prompt_source")
                    or "dataset_prompt_bank"
                ),
                modality=str(item.prompt_record.get("modality") or "text"),
                context_text=str(
                    item.prompt_record.get("context_text") or item.prompt_record.get("context") or ""
                ),
                media={str(k): v for k, v in item.media.items()},
                metadata=metadata,
            )
            # Completion order is durable output order. Stable prompt IDs and
            # source_index metadata preserve exact resume and audit identity.
            yield record
            source_queue.acknowledge(item.prompt_record_id)
            pending.remove(item)
            seen_text_hashes.add(text_hash)
            existing_prompt_ids.add(item.prompt_record_id)
            state["records"] = int(state["records"]) + 1
            state["estimated_tokens"] = int(state["estimated_tokens"]) + tok
            state["source_index"] = max(
                int(state.get("source_index", 0)), item.source_index + 1
            )
            state["last_record_id"] = record.id
            state["last_seconds"] = round(time.time() - item.started, 3)
            state["last_estimated_tokens"] = tok
            _commit_bucket_progress(state, item.prompt_record, tok)
            save_progress(progress_path, state)

            if target_estimated_tokens > 0 and int(state["estimated_tokens"]) >= target_estimated_tokens:
                reached_target = True
                break
            fill_window()
    finally:
        source_stop.set()
        for item in pending:
            item.future.cancel()
        executor.shutdown(wait=not reached_target and not pending, cancel_futures=True)
        producer.join(timeout=0.2)
        source_queue.requeue_inflight()
        source_queue.close()

    if target_estimated_tokens > 0 and int(state["estimated_tokens"]) < target_estimated_tokens:
        raise RuntimeError(
            f"teacher dataset underfilled: have {state['estimated_tokens']} estimated tokens, "
            f"need {target_estimated_tokens}; inspect the dataset source failure summary"
        )


def generate_records(
    cfg: TeacherConfig,
    prompt_records: Iterable[dict[str, Any]],
    target_estimated_tokens: int,
    progress_path: Path,
    output_path: Path | None = None,
    data_fingerprint: str = "",
    token_counter: Callable[[str], int] = estimate_tokens,
    prompt_limiter: Callable[[str, dict[str, Any]], str] = _identity_prompt_limiter,
    throughput: _ThroughputMeter | None = None,
) -> Iterable[TeacherSupervisedRecord]:
    if cfg.concurrency < 1:
        raise ValueError("teacher concurrency must be at least 1")
    if cfg.prefetch_records < 1:
        raise ValueError("teacher prefetch_records must be at least 1")
    implementation = _generate_records_sync if cfg.concurrency == 1 else _generate_records_concurrent
    yield from implementation(
        cfg,
        prompt_records,
        target_estimated_tokens,
        progress_path,
        output_path,
        data_fingerprint,
        token_counter,
        prompt_limiter,
        throughput,
    )


def _append_record_durable(path: Path, record: TeacherSupervisedRecord) -> None:
    record.validate_formal()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(asdict(record), ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(f"short teacher JSONL write: {written}/{len(payload)} bytes")
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=["lmstudio", "ollama", "llamacpp", "openai-compatible"], default="lmstudio")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--source-config", type=Path, default=Path("configs/dataset_sources.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("data/media_cache"))
    parser.add_argument("--max-prompt-chars", type=int, default=11000)
    parser.add_argument("--sources", default="")
    parser.add_argument("--max-records-per-source", type=int, default=0)
    parser.add_argument("--max-total-records", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/teacher_supervised/teacher_outputs.jsonl"))
    parser.add_argument("--progress", type=Path, default=Path("data/teacher_supervised/progress.json"))
    parser.add_argument("--target-estimated-tokens", type=int, default=0)
    parser.add_argument(
        "--max-tokens-per-sample",
        type=int,
        default=None,
        help="Optional teacher output cap; omit to let the runtime stop naturally.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("DG_TEACHER_MAX_RETRIES", "5")),
    )
    parser.add_argument("--retry-base-s", type=float, default=2.0)
    parser.add_argument("--min-estimated-tokens", type=int, default=None)
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("DG_STUDENT_MODEL", "google/gemma-4-E4B-it"),
    )
    parser.add_argument("--api-key-env", default="DG_TEACHER_API_KEY")
    parser.add_argument("--max-consecutive-failures", type=int, default=20)
    parser.add_argument(
        "--stream-max-retries",
        type=int,
        default=int(os.environ.get("DG_STREAM_MAX_RETRIES", "8")),
    )
    parser.add_argument(
        "--stream-retry-base-s",
        type=float,
        default=float(os.environ.get("DG_STREAM_RETRY_BASE_S", "2")),
    )
    parser.add_argument(
        "--stream-retry-max-s",
        type=float,
        default=float(os.environ.get("DG_STREAM_RETRY_MAX_S", "60")),
    )
    parser.add_argument(
        "--stream-checkpoint-rows",
        type=int,
        default=int(os.environ.get("DG_STREAM_CHECKPOINT_ROWS", "256")),
        help="Snapshot resumable HF stream state every N rows; 0 disables snapshots.",
    )
    parser.add_argument(
        "--stream-source-prefetch-records",
        type=int,
        default=int(os.environ.get("DG_STREAM_SOURCE_PREFETCH_RECORDS", "64")),
        help="Per-source background row buffer used to hide remote shard latency.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("DG_TEACHER_CONCURRENCY", "10")),
    )
    parser.add_argument(
        "--prefetch-records",
        type=int,
        default=int(os.environ.get("DG_TEACHER_PREFETCH_RECORDS", "16384")),
        help="Maximum number of streamed prompts buffered in the disk spool (default: 16384).",
    )
    parser.add_argument(
        "--prefetch-dir",
        type=Path,
        default=Path(os.environ.get("DG_TEACHER_PREFETCH_DIR", "data/teacher_supervised/prompt_spool")),
        help="Directory for persistent, resumable SQLite prompt spools.",
    )
    parser.add_argument(
        "--student-prefix-length",
        type=int,
        default=int(os.environ.get("DG_PREFIX_LENGTH", "2048")),
    )
    args = parser.parse_args()

    base_url = args.base_url
    if base_url is None:
        base_url = DEFAULT_OLLAMA_BASE_URL if args.runtime == "ollama" else DEFAULT_LMSTUDIO_BASE_URL

    source_config: dict[str, Any] | None = None
    mix_validation_config: dict[str, Any] | None = None
    if args.input_jsonl is not None:
        if not args.input_jsonl.is_file():
            raise FileNotFoundError(args.input_jsonl)
        data_fingerprint = _sha256_json(
            {
                "input_jsonl_sha256": _sha256_file(args.input_jsonl),
                "sources": args.sources,
                "max_total_records": args.max_total_records,
            }
        )
        prompt_records = iter_jsonl(args.input_jsonl)
        configured_min_tokens = 8
    else:
        source_config = json.loads(args.source_config.read_text(encoding="utf-8"))
        source_names = {item.strip() for item in args.sources.split(",") if item.strip()} or None
        selected_sources = [
            item
            for item in source_config.get("sources", [])
            if item.get("enabled", True)
            and (
                source_names is None
                or item.get("id") in source_names
                or item.get("name") in source_names
            )
        ]
        selected_buckets = {
            str(item.get("bucket") or "")
            for item in selected_sources
        }
        mix_validation_config = {
            **source_config,
            "recommended_mix": [
                item
                for item in source_config.get("recommended_mix", [])
                if str(item.get("bucket") or "") in selected_buckets
            ],
        }
        data_fingerprint = _sha256_json(
            {
                "source_config_sha256": _sha256_file(args.source_config),
                "sources": sorted(source_names or []),
                "max_prompt_chars": args.max_prompt_chars,
                "max_records_per_source": args.max_records_per_source,
                "max_total_records": args.max_total_records,
            }
        )
        prompt_records = iter_prompt_records(
            source_config,
            source_names=source_names,
            max_chars=args.max_prompt_chars,
            media_dir=args.media_dir,
            max_records_per_source=args.max_records_per_source,
            max_total_records=args.max_total_records,
            streaming_retry={
                "max_retries": args.stream_max_retries,
                "base_s": args.stream_retry_base_s,
                "max_s": args.stream_retry_max_s,
                "checkpoint_rows": args.stream_checkpoint_rows,
            },
            source_prefetch_records=args.stream_source_prefetch_records,
        )
        configured_min_tokens = int(source_config.get("teacher_output_filters", {}).get("min_estimated_tokens", 8))

    from transformers import AutoConfig, AutoTokenizer

    tokenizer_revision = ""
    try:
        accounting_config = AutoConfig.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
        tokenizer_revision = str(
            getattr(accounting_config, "_commit_hash", None) or ""
        )
    except Exception:
        pass
    accounting_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        revision=tokenizer_revision or None,
        fix_broken_tokenizers=True,
    )

    def exact_token_count(text: str) -> int:
        return len(accounting_tokenizer.encode(text, add_special_tokens=False))

    def limit_prompt_to_student_context(
        prompt: str,
        media: dict[str, Any],
    ) -> str:
        has_image = bool(
            _as_list(media.get("image")) or _as_list(media.get("images"))
        )
        has_audio = bool(
            _as_list(media.get("audio")) or _as_list(media.get("audios"))
        )
        # Reserve multimodal soft tokens and chat-template tokens inside the
        # same prefix budget the student receives during corruption/training.
        reserve = 1024 if has_audio else 384 if has_image else 64
        token_budget = args.student_prefix_length - reserve
        if token_budget <= 0:
            raise ValueError(
                "student prefix length is too small for the required chat/image reserve"
            )
        token_ids = accounting_tokenizer.encode(
            prompt, add_special_tokens=False
        )
        if len(token_ids) <= token_budget:
            return prompt
        return accounting_tokenizer.decode(
            token_ids[-token_budget:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    cfg = TeacherConfig(
        runtime=args.runtime,
        model=args.model,
        base_url=base_url,
        max_tokens=args.max_tokens_per_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        retry_base_s=args.retry_base_s,
        min_estimated_tokens=args.min_estimated_tokens or configured_min_tokens,
        tokenizer_name_or_path=args.tokenizer,
        tokenizer_revision=tokenizer_revision,
        api_key=os.environ.get(args.api_key_env, ""),
        max_consecutive_failures=args.max_consecutive_failures,
        concurrency=args.concurrency,
        prefetch_records=args.prefetch_records,
        prefetch_dir=args.prefetch_dir,
        student_prefix_length=args.student_prefix_length,
    )
    _repair_jsonl_tail(args.output)
    initial_state = read_progress(args.progress, output_path=args.output)
    initial_records = int(initial_state.get("records", 0))
    initial_tokens = int(initial_state.get("estimated_tokens", 0))

    if args.target_estimated_tokens > 0:
        pbar = tqdm(
            total=args.target_estimated_tokens,
            initial=min(initial_tokens, args.target_estimated_tokens),
            unit="tok",
            unit_scale=True,
            dynamic_ncols=True,
            desc="Generating teacher dataset",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        )
    elif args.max_total_records > 0:
        pbar = tqdm(
            total=args.max_total_records,
            initial=min(initial_records, args.max_total_records),
            unit="rec",
            dynamic_ncols=True,
            desc="Generating teacher dataset",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
        )
    else:
        pbar = tqdm(
            initial=initial_tokens,
            unit="tok",
            unit_scale=True,
            dynamic_ncols=True,
            desc="Generating teacher dataset",
            bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}{postfix}]",
        )

    written = 0
    current_tokens = initial_tokens
    throughput = _ThroughputMeter()
    progress_lock = threading.Lock()
    progress_stop = threading.Event()
    progress_fields: dict[str, Any] = {
        "records": initial_records,
        "total_tps": "warming",
    }
    if args.target_estimated_tokens > 0:
        progress_fields["eta"] = "?"

    def refresh_throughput() -> None:
        while not progress_stop.wait(1.0):
            total_tps = throughput.total_tps()
            with progress_lock:
                progress_fields["total_tps"] = (
                    f"{total_tps:.1f}" if total_tps is not None else "warming"
                )
                if args.target_estimated_tokens > 0:
                    progress_fields["eta"] = _throughput_eta(
                        args.target_estimated_tokens,
                        current_tokens,
                        total_tps,
                    )
                pbar.set_postfix(progress_fields, refresh=True)

    progress_thread = threading.Thread(
        target=refresh_throughput,
        name="teacher-throughput-display",
        daemon=True,
    )
    progress_thread.start()
    try:
        for record in generate_records(
            cfg,
            prompt_records,
            args.target_estimated_tokens,
            args.progress,
            args.output,
            data_fingerprint=data_fingerprint,
            token_counter=exact_token_count,
            prompt_limiter=limit_prompt_to_student_context,
            throughput=throughput,
        ):
            _append_record_durable(args.output, record)
            written += 1
            current_tokens += record.estimated_tokens
            with progress_lock:
                progress_fields["records"] = initial_records + written
                progress_fields["last_tok"] = record.estimated_tokens
                if args.target_estimated_tokens > 0 or args.max_total_records == 0:
                    pbar.update(record.estimated_tokens)
                else:
                    progress_fields["tokens"] = f"{current_tokens:,}"
                    pbar.update(1)
                pbar.set_postfix(progress_fields)
    finally:
        progress_stop.set()
        progress_thread.join(timeout=2)
        pbar.close()
    final_state = read_progress(
        args.progress,
        output_path=args.output,
        expected_fingerprint=generation_fingerprint(cfg, data_fingerprint),
    )
    if mix_validation_config is not None:
        final_state["mix_validation"] = validate_generated_mix(
            final_state, mix_validation_config
        )
        save_progress(args.progress, final_state)
    print(json.dumps({"records_written": written, "output": str(args.output), "total": final_state}, indent=2))


if __name__ == "__main__":
    main()
