from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import hashlib
import io
import json
import math
import os
import queue
import re
import random
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse


class SourceMixError(RuntimeError):
    """Raised when the configured training mixture cannot be satisfied."""


class SourceStreamError(SourceMixError):
    """Raised when a dataset stream cannot recover without changing its data mix."""


class StreamPauseState:
    """A shared connectivity gate for every remote dataset source."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._paused: dict[str, str] = {}
        self._offline = False
        self._probing = False
        self._next_probe_at = 0.0

    def pause(self, source_id: str, error: BaseException, poll_s: float = 30.0) -> bool:
        """Close the global gate and return True only for the first notification."""
        with self._condition:
            first = not self._offline
            self._offline = True
            self._paused[source_id] = f"{type(error).__name__}: {error}"
            if first:
                self._next_probe_at = time.monotonic() + poll_s
            self._condition.notify_all()
            return first

    def wait_until_online(
        self,
        probe: Any,
        poll_s: float,
    ) -> bool:
        """Block all readers while one elected reader performs silent probes."""
        while True:
            with self._condition:
                if not self._offline:
                    return False
                now = time.monotonic()
                if self._probing:
                    self._condition.wait()
                    continue
                if now < self._next_probe_at:
                    self._condition.wait(timeout=self._next_probe_at - now)
                    continue
                self._probing = True

            try:
                online = bool(probe())
            except Exception:  # A connectivity probe must never kill generation.
                online = False

            with self._condition:
                self._probing = False
                if online:
                    self._offline = False
                    self._paused.clear()
                    self._next_probe_at = 0.0
                    self._condition.notify_all()
                    return True
                self._next_probe_at = time.monotonic() + poll_s
                self._condition.notify_all()

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._offline


def _hf_connectivity_probe(timeout_s: float = 3.0) -> bool:
    """Check DNS and TCP reachability without issuing a noisy Hub request."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    parsed = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
    host = parsed.hostname or "huggingface.co"
    port = parsed.port or (80 if parsed.scheme == "http" else 443)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _first_text(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = _as_text(row.get(field))
        if value:
            return value
    return ""


def _messages_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "conversation", "conversations", "dialogue", "turns"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _message_text(message: dict[str, Any]) -> str:
    for key in ("content", "value", "text", "message"):
        value = _as_text(message.get(key))
        if value:
            return value
    return ""


def _message_role(message: dict[str, Any]) -> str:
    return _as_text(message.get("role") or message.get("from") or message.get("speaker")).lower()


def _extract_user_prompt(row: dict[str, Any]) -> str:
    messages = _messages_from_row(row)
    for message in messages:
        if _message_role(message) in {"user", "human", "prompter", "client"}:
            text = _message_text(message)
            if text:
                return text
    if messages:
        return _message_text(messages[0])
    return _first_text(row, ["prompt", "instruction", "question", "query", "input", "user", "title"])


def _extract_context(row: dict[str, Any]) -> str:
    context = _first_text(
        row,
        [
            "context",
            "document",
            "passage",
            "article",
            "text",
            "content",
            "body",
            "code",
            "hint",
            "lecture",
        ],
    )
    if context:
        return context
    messages = _messages_from_row(row)
    user_turns = [_message_text(msg) for msg in messages if _message_role(msg) in {"user", "human", "prompter", "client"}]
    return "\n\n".join(turn for turn in user_turns[:3] if turn)


def _clean_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    return (clipped.rsplit(" ", 1)[0] or clipped).strip()


def _strip_media_placeholders(text: str) -> str:
    """Remove dataset-side image markers; the real image is sent separately."""

    return re.sub(
        r"(?i)(?:<\s*image\s*>|\[\s*image\s*\]|\{\s*image\s*\})",
        " ",
        text,
    ).strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k not in {"array", "bytes"}}
    return str(value)


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_row_key(source: dict[str, Any], row: dict[str, Any], prompt: str, context: str) -> str:
    explicit = _first_text(
        row,
        ["id", "uuid", "conversation_id", "question_id", "doc_id", "document_id", "url", "__index_level_0__"],
    )
    return _stable_digest(
        {
            "source": source["id"],
            "config": source.get("config"),
            "split": source.get("split", "train"),
            "upstream_id": explicit,
            "prompt": prompt,
            "context": context,
        }
    )


def _save_image(value: Any, path: Path) -> str | None:
    if hasattr(value, "save"):
        path.parent.mkdir(parents=True, exist_ok=True)
        value.save(path)
        return str(path)
    if isinstance(value, dict):
        if value.get("path"):
            return str(value["path"])
        if value.get("bytes"):
            try:
                from PIL import Image
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("Pillow is required to materialize image bytes from datasets.") from exc
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(value["bytes"])).save(path)
            return str(path)
    if isinstance(value, (str, Path)):
        return str(value)
    return None


def _save_audio(value: Any, path: Path) -> dict[str, Any] | str | None:
    if isinstance(value, dict):
        if value.get("path"):
            return str(value["path"])
        if "array" in value:
            try:
                import soundfile as sf
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("soundfile is required to materialize audio arrays from datasets.") from exc
            sampling_rate = int(value.get("sampling_rate") or 16000)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, value["array"], sampling_rate)
            return {"path": str(path), "sampling_rate": sampling_rate}
    if isinstance(value, (str, Path)):
        return str(value)
    return None


def _collect_media(source: dict[str, Any], row: dict[str, Any], media_dir: Path | None, record_key: str) -> dict[str, Any]:
    media: dict[str, Any] = {}
    if isinstance(row.get("media"), dict):
        media.update(_jsonable(row["media"]))
    source_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source.get("name") or source["id"]))

    def media_path(kind: str, index: int, suffix: str) -> Path:
        if media_dir is None:
            raise ValueError("media_dir is required to materialize decoded media objects")
        return media_dir / source_slug / f"{record_key}_{kind}_{index}{suffix}"

    for key in ("image", "images"):
        if key not in row:
            continue
        values = row[key] if isinstance(row[key], list) else [row[key]]
        paths = [_save_image(value, media_path("image", i, ".png")) for i, value in enumerate(values)]
        paths = [path for path in paths if path]
        if paths:
            media["images"] = paths

    # The E4B student preserves its audio tower. The production source stays
    # disabled until a language config/split and an audio-capable teacher
    # endpoint are explicitly verified.
    for key in ("audio", "audios"):
        if key not in row:
            continue
        values = row[key] if isinstance(row[key], list) else [row[key]]
        paths = [_save_audio(value, media_path("audio", i, ".wav")) for i, value in enumerate(values)]
        paths = [path for path in paths if path]
        if paths:
            media["audio"] = paths

    for key in ("media_path", "url"):
        value = row.get(key)
        if isinstance(value, list):
            media[key] = [_jsonable(item) for item in value]
        elif value:
            media[key] = _jsonable(value)
    return media


def _prompt_record(source: dict[str, Any], row: dict[str, Any], max_chars: int, media_dir: Path | None = None) -> dict[str, Any] | None:
    modality = str(source.get("modality") or "text")
    source_id = str(source["id"])
    role = str(source.get("role") or "prompt_context_bank")
    bucket = str(source.get("bucket") or ("image" if "image" in modality else "text"))
    prompt = _clean_text(_extract_user_prompt(row), max_chars)
    context = _clean_text(_extract_context(row), max_chars)
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        option_lines = [
            f"{chr(65 + index)}. {_as_text(choice)}"
            for index, choice in enumerate(choices[:26])
            if _as_text(choice)
        ]
        if option_lines:
            prompt = _clean_text(
                prompt + "\nOptions:\n" + "\n".join(option_lines),
                max_chars,
            )

    if modality == "image":
        prompt = _clean_text(_strip_media_placeholders(prompt), max_chars)
        context = _clean_text(_strip_media_placeholders(context), max_chars)
        if not prompt:
            prompt = str(
                source.get("teacher_prompt")
                or "Describe and analyze the image carefully, including the important visual details."
            )

    if role.endswith("_warmup_or_context_seed") or role in {
        "plain_text_corpus",
        "code_context_stream",
        "math_science_context_stream",
    }:
        if not context:
            return None
        prompt = str(
            source.get("teacher_prompt")
            or "Respond helpfully using the provided context while preserving the teacher model's normal style."
        )
    if not prompt and not context:
        return None

    # max_chars is a teacher-request budget, not a per-field budget. Without
    # this final combined clamp a long context plus a long question could be
    # almost 2x the advertised limit and repeatedly exceed vLLM max_model_len.
    prompt = _clean_text(prompt, max_chars)
    if context and prompt and context != prompt:
        context_budget = max(0, max_chars - len(prompt) - 32)
        context = _clean_text(context, context_budget)
    else:
        context = _clean_text(context, max_chars)

    row_key = _source_row_key(source, row, prompt, context)
    media = _collect_media(source, row, media_dir, row_key)
    if modality == "image" and not (
        _as_text(media.get("image"))
        or (
            isinstance(media.get("images"), list)
            and any(_as_text(value) for value in media["images"])
        )
    ):
        return None
    return {
        "id": f"prompt-{row_key}",
        "source": source_id,
        "source_name": str(source.get("name") or source_id),
        "bucket": bucket,
        "role": role,
        "modality": modality,
        "prompt_text": prompt,
        "context_text": context if context != prompt else "",
        "media": media,
        "metadata": {
            "dataset": source_id,
            "dataset_config": source.get("config"),
            "dataset_split": source.get("split", "train"),
            "upstream_record_id": _first_text(
                row, ["id", "uuid", "conversation_id", "question_id", "doc_id", "document_id", "url"]
            ),
            "source_record_fingerprint": row_key,
            "source_name": str(source.get("name") or source_id),
            "bucket": bucket,
            "license_hint": str(source.get("license_hint") or ""),
            "target_policy": "teacher_generated_output_only",
        },
    }


def _open_hf_dataset(source: dict[str, Any]) -> Any:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Install the `datasets` package to prepare prompt/context banks.") from exc
    kwargs = {
        "split": source.get("split", "train"),
        "streaming": bool(source.get("streaming", True)),
        "trust_remote_code": bool(source.get("trust_remote_code", False)),
    }
    config = source.get("config")
    return load_dataset(source["id"], config, **kwargs) if config else load_dataset(source["id"], **kwargs)


_STREAM_RETRY_CONFIG_LOCK = threading.Lock()


def _configure_streaming_network_retries() -> None:
    """Route network failures directly to our offline-pause policy.

    Hugging Face Hub and datasets both install their own retry loops.  The
    wrappers are process-wide because dataset reads happen concurrently; a
    temporary monkey-patch can be restored by one source while another source
    is still reading.
    """

    with _STREAM_RETRY_CONFIG_LOCK:
        try:
            from datasets.utils import file_utils as datasets_file_utils

            # datasets wraps every streaming file object's read() with 20
            # retries. Bypass that wrapper so the original read still happens
            # once and any failure reaches this iterator immediately.
            add_read_retries = getattr(
                datasets_file_utils, "_add_retries_to_file_obj_read_method", None
            )
            if callable(add_read_retries) and not getattr(
                add_read_retries, "_dg_no_retry", False
            ):
                def leave_read_unwrapped(file_obj: Any) -> Any:
                    return file_obj

                leave_read_unwrapped._dg_no_retry = True  # type: ignore[attr-defined]
                datasets_file_utils._add_retries_to_file_obj_read_method = (
                    leave_read_unwrapped
                )
        except Exception:  # noqa: BLE001
            pass

        try:
            from huggingface_hub import file_download, hf_file_system
        except Exception:  # noqa: BLE001
            return

        def patch_http_get(module: Any) -> None:
            original = getattr(module, "http_get", None)
            if not callable(original) or getattr(original, "_dg_no_retry", False):
                return

            def http_get_no_retry(*args: Any, **kwargs: Any) -> Any:
                kwargs["_nb_retries"] = 0
                return original(*args, **kwargs)

            http_get_no_retry._dg_no_retry = True  # type: ignore[attr-defined]
            module.http_get = http_get_no_retry

        def patch_http_stream_backoff(module: Any) -> None:
            original = getattr(module, "http_stream_backoff", None)
            if not callable(original) or getattr(original, "_dg_no_retry", False):
                return

            @contextlib.contextmanager
            def http_stream_backoff_no_retry(*args: Any, **kwargs: Any) -> Iterator[Any]:
                kwargs["max_retries"] = 0
                with original(*args, **kwargs) as response:
                    yield response

            http_stream_backoff_no_retry._dg_no_retry = True  # type: ignore[attr-defined]
            module.http_stream_backoff = http_stream_backoff_no_retry

        patch_http_get(file_download)
        # Streaming parquet reads use the function imported directly into
        # hf_file_system, while regular downloads use file_download's copy.
        patch_http_stream_backoff(file_download)
        patch_http_stream_backoff(hf_file_system)


def _http_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_transient_stream_error(exc: BaseException) -> bool:
    status = _http_status(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in {
            errno.ECONNABORTED,
            errno.ECONNREFUSED,
            errno.ECONNRESET,
            errno.EHOSTUNREACH,
            errno.ENETDOWN,
            errno.ENETUNREACH,
            errno.ETIMEDOUT,
        }
    name = type(exc).__name__.lower()
    return any(token in name for token in ("connection", "timeout", "ratelimit"))


def _is_offline_stream_error(exc: BaseException) -> bool:
    """Return whether retry limits should be suspended until connectivity returns."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _http_status(current) is not None:
            return False
        if isinstance(current, OSError) and current.errno in {
            errno.EHOSTUNREACH,
            errno.ENETDOWN,
            errno.ENETUNREACH,
        }:
            return True
        name = type(current).__name__.lower()
        message = str(current).lower()
        if "gaierror" in name or any(
            token in message
            for token in (
                "network is unreachable",
                "no route to host",
                "name or service not known",
                "nodename nor servname provided",
                "temporary failure in name resolution",
                "failed to resolve",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class _ResilientDatasetIterator:
    def __init__(self, source: dict[str, Any], retry_config: dict[str, Any] | None = None):
        _configure_streaming_network_retries()
        self.source = source
        self.rows_read = 0
        self.dataset: Any | None = None
        self.iterator: Iterator[dict[str, Any]] | None = None
        self.resume_state: dict[str, Any] | None = None
        self.resume_rows = 0
        retry_config = retry_config or {}
        self.max_retries = int(retry_config.get("max_retries", 8))
        self.retry_base_s = float(retry_config.get("base_s", 2.0))
        self.retry_max_s = float(retry_config.get("max_s", 60.0))
        self.checkpoint_rows = int(retry_config.get("checkpoint_rows", 256))
        self.offline_poll_s = float(retry_config.get("offline_poll_s", 30.0))
        self.pause_state = retry_config.get("pause_state") or StreamPauseState()
        self.connectivity_probe = retry_config.get("connectivity_probe") or _hf_connectivity_probe
        if (
            self.max_retries < 0
            or self.retry_base_s < 0
            or self.retry_max_s < 0
            or self.checkpoint_rows < 0
            or self.offline_poll_s < 0
        ):
            raise ValueError("stream retry settings must be non-negative")

    def __iter__(self) -> _ResilientDatasetIterator:
        return self

    def _reopen(self) -> None:
        dataset = _open_hf_dataset(self.source)
        restored_rows = 0
        can_snapshot = callable(getattr(dataset, "state_dict", None)) and callable(
            getattr(dataset, "load_state_dict", None)
        )
        if self.resume_state is not None:
            restore = getattr(dataset, "load_state_dict", None)
            if callable(restore):
                try:
                    restore(copy.deepcopy(self.resume_state))
                    restored_rows = self.resume_rows
                except Exception:  # Fall back to the portable skip path below.
                    dataset = _open_hf_dataset(self.source)
                    can_snapshot = callable(getattr(dataset, "state_dict", None)) and callable(
                        getattr(dataset, "load_state_dict", None)
                    )
                    restored_rows = 0
        rows_to_skip = self.rows_read - restored_rows
        if rows_to_skip and can_snapshot:
            # Replay at most checkpoint_rows-1 records on the unchanged dataset
            # pipeline. Applying dataset.skip() here would wrap the pipeline,
            # making its next state_dict incompatible with a freshly opened
            # dataset on a later retry.
            iterator = iter(dataset)
            for _ in range(rows_to_skip):
                next(iterator)
            self.dataset = dataset
            self.iterator = iterator
            return
        if rows_to_skip:
            skip = getattr(dataset, "skip", None)
            if not callable(skip):
                raise SourceStreamError(
                    f"streaming dataset {self.source['id']} cannot resume: iterator has no skip()"
                )
            dataset = skip(rows_to_skip)
        self.dataset = dataset
        self.iterator = iter(dataset)

    def _checkpoint(self) -> None:
        if self.checkpoint_rows <= 0 or self.rows_read % self.checkpoint_rows:
            return
        state_dict = getattr(self.dataset, "state_dict", None)
        if not callable(state_dict):
            return
        try:
            self.resume_state = copy.deepcopy(state_dict())
            self.resume_rows = self.rows_read
        except Exception:
            # Checkpointing is an optimization. The exact but slower skip-based
            # recovery path remains available when a dataset cannot snapshot.
            self.resume_state = None
            self.resume_rows = 0

    def __next__(self) -> dict[str, Any]:
        errors: list[str] = []
        attempt = 0
        source_id = str(self.source["id"])
        while attempt <= self.max_retries:
            try:
                if (
                    isinstance(self.pause_state, StreamPauseState)
                    and self.pause_state.wait_until_online(
                        self.connectivity_probe, self.offline_poll_s
                    )
                ):
                    print(
                        "network restored; resuming all streaming datasets",
                        file=sys.stderr,
                    )
                if self.iterator is None:
                    self._reopen()
                assert self.iterator is not None
                row = next(self.iterator)
                self.rows_read += 1
                self._checkpoint()
                return row
            except StopIteration:
                raise
            except Exception as exc:  # noqa: BLE001
                self.dataset = None
                self.iterator = None
                if _is_offline_stream_error(exc):
                    first_pause = False
                    if isinstance(self.pause_state, StreamPauseState):
                        first_pause = self.pause_state.pause(
                            source_id, exc, self.offline_poll_s
                        )
                    if first_pause:
                        print(
                            "network unavailable; pausing all streaming datasets. "
                            "Disk-spooled prompts remain available; a single silent "
                            f"connectivity probe will run every {self.offline_poll_s:g}s",
                            file=sys.stderr,
                        )
                    # Connectivity failures are an availability pause, not a
                    # dataset failure. They must not consume the retry budget
                    # or let callers quarantine an otherwise valid source.
                    continue
                if not _is_transient_stream_error(exc):
                    raise SourceStreamError(
                        f"streaming dataset {self.source['id']} failed permanently at row "
                        f"{self.rows_read}: {type(exc).__name__}: {exc}"
                    ) from exc
                errors.append(f"attempt={attempt + 1}: {type(exc).__name__}: {exc}")
                if attempt >= self.max_retries:
                    break
                ceiling = min(self.retry_max_s, self.retry_base_s * (2**attempt))
                delay = random.uniform(0.5 * ceiling, ceiling) if ceiling > 0 else 0
                print(
                    f"streaming dataset {self.source['id']} interrupted at row {self.rows_read}; "
                    f"retrying in {delay:.1f}s ({attempt + 1}/{self.max_retries})",
                    file=sys.stderr,
                )
                time.sleep(delay)
                attempt += 1
        raise SourceStreamError(
            f"streaming dataset {self.source['id']} exhausted {self.max_retries} retries at "
            f"row {self.rows_read}; refusing to silently remove the source: {' | '.join(errors)}"
        )


def _load_dataset_iter(
    source: dict[str, Any], retry_config: dict[str, Any] | None = None
) -> Iterable[dict[str, Any]]:
    if bool(source.get("streaming", True)):
        return _ResilientDatasetIterator(source, retry_config)
    return iter(_open_hf_dataset(source))


def _load_manifest_iter(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    manifest_path = source.get("manifest_path")
    if manifest_path is None:
        name = source.get("name") or re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source["id"]))
        manifest_path = Path("data/manifests") / f"{name}.jsonl"
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"{source['id']} is marked manifest_only, but manifest file is missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _bounded_limit(source: dict[str, Any], cli_limit: int) -> int:
    configured = int(source.get("max_records") or 0)
    positive = [value for value in (configured, int(cli_limit or 0)) if value > 0]
    return min(positive) if positive else 0


def _training_weights(config: dict[str, Any], sources: list[dict[str, Any]]) -> tuple[list[str], dict[str, float], set[str]]:
    enabled_buckets = {
        str(source.get("bucket") or ("image" if "image" in str(source.get("modality")) else "text"))
        for source in sources
    }
    mix = config.get("recommended_mix") or config.get("training_mix") or []
    order: list[str] = []
    weights: dict[str, float] = {}
    required: set[str] = set()
    for item in mix:
        bucket = str(item.get("bucket") or "")
        if not bucket or bucket == "eval_only" or item.get("training") is False:
            continue
        share = float(item.get("share") or 0.0)
        if share <= 0:
            continue
        order.append(bucket)
        weights[bucket] = share
        if item.get("required", True):
            required.add(bucket)
    if weights:
        total = sum(weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise SourceMixError(f"active training bucket shares must sum to 1.0, got {total:.12f}")
        missing = enabled_buckets - set(weights)
        if missing:
            raise SourceMixError(f"enabled sources reference buckets absent from recommended_mix: {sorted(missing)}")
        selected = [bucket for bucket in order if bucket in enabled_buckets]
        selected_total = sum(weights[bucket] for bucket in selected)
        if selected and not math.isclose(selected_total, 1.0, abs_tol=1e-9):
            weights = {bucket: weights[bucket] / selected_total for bucket in selected}
            required &= set(selected)
        order = selected
    else:
        order = sorted(enabled_buckets)
        weights = {bucket: 1.0 / len(order) for bucket in order} if order else {}
        required = set(order)
    return order, weights, required


def _integer_quotas(total: int, order: list[str], weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {}
    raw = {bucket: total * weights[bucket] for bucket in order}
    quotas = {bucket: int(math.floor(value)) for bucket, value in raw.items()}
    remaining = total - sum(quotas.values())
    ranked = sorted(order, key=lambda bucket: (raw[bucket] - quotas[bucket], weights[bucket]), reverse=True)
    for bucket in ranked[:remaining]:
        quotas[bucket] += 1
    return quotas


@dataclass
class _SourceState:
    source: dict[str, Any]
    rows: Iterator[dict[str, Any]]
    limit: int
    yielded: int = 0
    rejected: int = 0
    rows_read: int = 0


@dataclass
class _PrefetchFailure:
    error: BaseException


_PREFETCH_END = object()


class _BackgroundPrefetchIterator:
    """Read one remote dataset ahead without letting it block other sources."""

    def __init__(self, rows: Iterator[dict[str, Any]], max_records: int, name: str):
        self._rows = rows
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_records)
        self._stop = threading.Event()
        self._done = False
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)[:48]
        self._thread = threading.Thread(
            target=self._run,
            name=f"dataset-prefetch-{safe_name}",
            daemon=True,
        )
        self._thread.start()

    def __iter__(self) -> _BackgroundPrefetchIterator:
        return self

    def _publish(self, value: Any) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        terminal: Any = _PREFETCH_END
        try:
            while not self._stop.is_set():
                if not self._publish(next(self._rows)):
                    return
        except StopIteration:
            pass
        except BaseException as exc:  # Re-raised by the consuming thread.
            terminal = _PrefetchFailure(exc)
        finally:
            self._publish(terminal)
            close = getattr(self._rows, "close", None)
            if callable(close):
                close()

    def next(self, timeout: float | None = None) -> dict[str, Any]:
        if self._done:
            raise StopIteration
        value = self._queue.get(timeout=timeout)
        if value is _PREFETCH_END:
            self._done = True
            raise StopIteration
        if isinstance(value, _PrefetchFailure):
            self._done = True
            raise value.error
        return value

    def __next__(self) -> dict[str, Any]:
        return self.next()

    def close(self) -> None:
        self._stop.set()


def iter_prompt_records(
    config: dict[str, Any],
    source_names: set[str] | None,
    max_chars: int,
    media_dir: Path | None,
    max_records_per_source: int = 0,
    max_total_records: int = 0,
    streaming_retry: dict[str, Any] | None = None,
    source_prefetch_records: int = 64,
) -> Iterable[dict[str, Any]]:
    if source_prefetch_records < 0:
        raise ValueError("source_prefetch_records must be non-negative")
    sources = [
        src
        for src in config["sources"]
        if src.get("enabled", True) and not src.get("eval_only", False) and str(src.get("bucket")) != "eval_only"
    ]
    if source_names:
        sources = [src for src in sources if src["id"] in source_names or src.get("name") in source_names]
    if not sources:
        raise SourceMixError("No enabled training sources were selected.")
    order, weights, required_buckets = _training_weights(config, sources)
    by_bucket: dict[str, list[_SourceState]] = {bucket: [] for bucket in order}
    failures: list[dict[str, str]] = []
    streaming_retry = dict(streaming_retry or {})
    all_states: list[_SourceState] = []
    for source in sources:
        bucket = str(source.get("bucket") or ("image" if "image" in str(source.get("modality")) else "text"))
        try:
            rows = (
                _load_manifest_iter(source)
                if source.get("manifest_only")
                else _load_dataset_iter(source, streaming_retry)
            )
            row_iterator: Iterator[dict[str, Any]] = iter(rows)
            if (
                source_prefetch_records > 0
                and bool(source.get("streaming", True))
                and not source.get("manifest_only")
            ):
                # Decoded image/audio rows can be large, so keep only a couple
                # in memory while allowing lightweight text sources to read far
                # enough ahead to hide shard-open and network latency.
                modality = str(source.get("modality") or "text")
                per_source_records = (
                    min(source_prefetch_records, 2)
                    if modality in {"image", "audio"}
                    else source_prefetch_records
                )
                row_iterator = _BackgroundPrefetchIterator(
                    row_iterator,
                    max_records=max(1, per_source_records),
                    name=str(source.get("name") or source["id"]),
                )
            state = _SourceState(source, row_iterator, _bounded_limit(source, max_records_per_source))
            by_bucket[bucket].append(state)
            all_states.append(state)
        except Exception as exc:  # noqa: BLE001
            raise SourceMixError(
                f"failed to open dataset source {source.get('id')}; refusing to alter the configured mix: {exc}"
            ) from exc

    requested_total = int(max_total_records or 0)
    quotas = _integer_quotas(requested_total, order, weights)
    emitted = {bucket: 0 for bucket in order}
    source_cursor = {bucket: 0 for bucket in order}
    total_records = 0

    def next_from_bucket(bucket: str) -> dict[str, Any] | None:
        states = by_bucket[bucket]
        while states:
            cursor = source_cursor[bucket] % len(states)
            state = states[cursor]
            source_cursor[bucket] = cursor + 1
            if state.limit > 0 and state.yielded >= state.limit:
                close = getattr(state.rows, "close", None)
                if callable(close):
                    close()
                states.pop(cursor)
                if states:
                    source_cursor[bucket] %= len(states)
                continue
            try:
                if isinstance(state.rows, _BackgroundPrefetchIterator):
                    # Do not let one source opening a shard hold up ready
                    # sources in the same bucket. Prompt IDs, rather than these
                    # timing-dependent positions, are the durable resume key.
                    row = state.rows.next(timeout=0.05 if len(states) > 1 else None)
                else:
                    row = next(state.rows)
                state.rows_read += 1
            except queue.Empty:
                continue
            except StopIteration:
                close = getattr(state.rows, "close", None)
                if callable(close):
                    close()
                states.pop(cursor)
                if states:
                    source_cursor[bucket] %= len(states)
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append({"source": str(state.source.get("id")), "phase": "iterate", "error": repr(exc)})
                # An explicitly optional source that is unavailable before its
                # first row (for example, an unapproved gated dataset) may be
                # quarantined without changing the configured bucket quota.
                # Other sources in the same bucket must still satisfy it.
                if state.rows_read == 0 and not bool(state.source.get("required", False)):
                    source_id = str(state.source.get("id"))
                    close = getattr(state.rows, "close", None)
                    if callable(close):
                        close()
                    states.pop(cursor)
                    if states:
                        source_cursor[bucket] %= len(states)
                    print(
                        f"WARNING: quarantining optional dataset source {source_id} before its "
                        f"first row; bucket {bucket} quota remains unchanged: {exc}",
                        file=sys.stderr,
                    )
                    if not states and bucket in required_buckets:
                        raise SourceMixError(
                            f"optional source {source_id} is unavailable and no source remains "
                            f"for required bucket {bucket}: {exc}"
                        ) from exc
                    continue
                raise SourceMixError(
                    f"dataset source {state.source.get('id')} failed during iteration after "
                    f"{state.rows_read} rows; refusing to silently remove it from bucket {bucket}: {exc}"
                ) from exc
            try:
                record = _prompt_record(state.source, row, max_chars=max_chars, media_dir=media_dir)
            except Exception as exc:  # noqa: BLE001
                failures.append({"source": str(state.source.get("id")), "phase": "convert", "error": repr(exc)})
                state.rejected += 1
                continue
            if record is None:
                state.rejected += 1
                continue
            state.yielded += 1
            return record
        return None

    try:
        while True:
            eligible = [
                bucket
                for bucket in order
                if by_bucket[bucket] and (not quotas or emitted[bucket] < quotas[bucket])
            ]
            if not eligible:
                break
            bucket = min(eligible, key=lambda name: (emitted[name] / weights[name], order.index(name)))
            record = next_from_bucket(bucket)
            if record is None:
                continue
            yield record
            emitted[bucket] += 1
            total_records += 1
            if requested_total > 0 and total_records >= requested_total:
                break
    finally:
        for state in all_states:
            close = getattr(state.rows, "close", None)
            if callable(close):
                close()
        summary = {
            "dataset_mix_summary": {
                "requested_records": requested_total,
                "emitted_records": total_records,
                "bucket_weights": weights,
                "bucket_quotas": quotas,
                "by_bucket": emitted,
                "by_source": {
                    str(state.source["id"]): {
                        "yielded": state.yielded,
                        "rows_read": state.rows_read,
                        "rejected": state.rejected,
                        "required": bool(state.source.get("required", False)),
                    }
                    for state in all_states
                },
                "failures": failures,
            }
        }
        print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)

    unmet = {
        bucket: {"required": quotas.get(bucket, 1), "actual": emitted[bucket]}
        for bucket in required_buckets
        if (quotas and emitted[bucket] < quotas[bucket]) or (not quotas and emitted[bucket] == 0)
    }
    required_source_failures = [
        str(source["id"])
        for source in sources
        if source.get("required", False)
        and not any(state.source["id"] == source["id"] and state.yielded > 0 for state in all_states)
    ]
    if requested_total > 0 and total_records != requested_total:
        raise SourceMixError(f"dataset mixture underfilled: requested {requested_total}, emitted {total_records}; unmet={unmet}")
    if unmet or required_source_failures:
        raise SourceMixError(
            f"required dataset quota unmet: buckets={unmet}, sources={required_source_failures}, failures={failures}"
        )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    by_source: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                source = str(row["source"])
                bucket = str(row.get("bucket") or "unknown")
                by_source[source] = by_source.get(source, 0) + 1
                by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return {"records": count, "output": str(path), "by_source": by_source, "by_bucket": by_bucket}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_sources.json"))
    parser.add_argument("--output", type=Path, default=Path("data/prompt_context/prompts.jsonl"))
    parser.add_argument("--max-chars", type=int, default=11000)
    parser.add_argument("--media-dir", type=Path, default=Path("data/media_cache"))
    parser.add_argument("--sources", default="")
    parser.add_argument("--max-records-per-source", type=int, default=0)
    parser.add_argument("--max-total-records", type=int, default=0)
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
        "--stream-offline-poll-s",
        type=float,
        default=float(os.environ.get("DG_STREAM_OFFLINE_POLL_S", "30")),
        help="Seconds between connectivity checks while a dataset stream is offline.",
    )
    parser.add_argument(
        "--stream-checkpoint-rows",
        type=int,
        default=int(os.environ.get("DG_STREAM_CHECKPOINT_ROWS", "256")),
    )
    parser.add_argument(
        "--stream-source-prefetch-records",
        type=int,
        default=int(os.environ.get("DG_STREAM_SOURCE_PREFETCH_RECORDS", "64")),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_names = {item.strip() for item in args.sources.split(",") if item.strip()} or None
    rows = iter_prompt_records(
        config,
        source_names=source_names,
        max_chars=args.max_chars,
        media_dir=args.media_dir,
        max_records_per_source=args.max_records_per_source,
        max_total_records=args.max_total_records,
        streaming_retry={
            "max_retries": args.stream_max_retries,
            "base_s": args.stream_retry_base_s,
            "max_s": args.stream_retry_max_s,
            "checkpoint_rows": args.stream_checkpoint_rows,
            "offline_poll_s": args.stream_offline_poll_s,
        },
        source_prefetch_records=args.stream_source_prefetch_records,
    )
    result = write_jsonl(args.output, rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
