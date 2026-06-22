from __future__ import annotations

import argparse
import base64
import mimetypes
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from .constants import (
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_OLLAMA_BASE_URL,
)
from .data_contract import TeacherSupervisedRecord, iter_jsonl, write_teacher_jsonl
from .data_sources import iter_prompt_records


@dataclass
class TeacherConfig:
    runtime: str
    model: str
    base_url: str
    max_tokens: int
    temperature: float
    top_p: float
    timeout_s: int = 600


class TeacherClient:
    def generate(self, prompt: str, media: dict[str, Any] | None = None) -> str:
        raise NotImplementedError


class OpenAICompletionsClient(TeacherClient):
    """Prompted completions against LM Studio or llama.cpp OpenAI-compatible APIs."""

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def generate(self, prompt: str, media: dict[str, Any] | None = None) -> str:
        if media:
            return self._generate_chat(prompt, media)
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "stream": False,
        }
        response = self.session.post(
            f"{self.cfg.base_url.rstrip('/')}/completions",
            json=payload,
            timeout=self.cfg.timeout_s,
        )
        if response.status_code >= 400:
            return self._generate_chat(prompt, media=None)
        data = response.json()
        return data["choices"][0].get("text", "")

    def _generate_chat(self, prompt: str, media: dict[str, Any] | None) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": _chat_content(prompt, media or {})}],
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "stream": False,
        }
        response = self.session.post(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=self.cfg.timeout_s,
        )
        if response.status_code >= 400:
            raise requests.HTTPError(
                f"OpenAI-compatible teacher completion failed: {response.status_code} {response.text}",
                response=response,
            )
        data = response.json()
        return data["choices"][0]["message"].get("content", "")


class OllamaGenerateClient(TeacherClient):
    """Prompted Ollama generation."""

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def generate(self, prompt: str, media: dict[str, Any] | None = None) -> str:
        if media:
            raise RuntimeError("Ollama teacher generation does not support multimodal media in this pipeline.")
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "num_predict": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
            },
        }
        response = self.session.post(
            f"{self.cfg.base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=self.cfg.timeout_s,
        )
        response.raise_for_status()
        return response.json().get("response", "")


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
    # Runtime generation APIs do not always return token counts. This is only for progress;
    # formal token accounting is done by tokenize_blocks.py with the Gemma tokenizer.
    return max(1, len(text.encode("utf-8")) // 4)


def read_progress(progress_path: Path) -> dict[str, Any]:
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"records": 0, "estimated_tokens": 0}


def save_progress(progress_path: Path, state: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _build_prompt(record: dict[str, Any]) -> str:
    prompt = str(record.get("prompt_text") or record.get("prompt") or "").strip()
    context = str(record.get("context_text") or record.get("context") or "").strip()
    if context and prompt:
        return f"{context}\n\n{prompt}"
    if context:
        return context
    if prompt:
        return prompt
    raise ValueError("prompt record must contain prompt_text/prompt or context_text/context")


def generate_records(
    cfg: TeacherConfig,
    prompt_records: Iterable[dict[str, Any]],
    target_estimated_tokens: int,
    progress_path: Path,
) -> Iterable[TeacherSupervisedRecord]:
    client = make_client(cfg)
    state = read_progress(progress_path)
    if state.get("records") is None:
        state["records"] = 0
    for source_index, prompt_record in enumerate(prompt_records):
        if source_index < int(state.get("source_index", 0)):
            continue
        if target_estimated_tokens > 0 and state["estimated_tokens"] >= target_estimated_tokens:
            break
        started = time.time()
        prompt = _build_prompt(prompt_record)
        media = dict(prompt_record.get("media") or {})
        text = client.generate(prompt, media=media)
        tok = estimate_tokens(text)
        record = TeacherSupervisedRecord(
            id=str(uuid.uuid4()),
            source_model=cfg.model,
            runtime=cfg.runtime,
            prompt_text=prompt,
            text=text,
            estimated_tokens=tok,
            prompt_source=str(prompt_record.get("source") or prompt_record.get("prompt_source") or "dataset_prompt_bank"),
            modality=str(prompt_record.get("modality") or "text"),
            context_text=str(prompt_record.get("context_text") or prompt_record.get("context") or ""),
            media={str(k): v for k, v in media.items()},
            metadata={str(k): v for k, v in dict(prompt_record.get("metadata") or {}).items()},
        )
        state["records"] += 1
        state["estimated_tokens"] += tok
        state["source_index"] = source_index + 1
        state["last_record_id"] = record.id
        state["last_seconds"] = round(time.time() - started, 3)
        state["last_estimated_tokens"] = tok
        save_progress(progress_path, state)
        yield record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=["lmstudio", "ollama", "llamacpp", "openai-compatible"], default="lmstudio")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--source-config", type=Path, default=Path("configs/dataset_sources.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("data/media_cache"))
    parser.add_argument("--max-prompt-chars", type=int, default=12000)
    parser.add_argument("--sources", default="")
    parser.add_argument("--max-records-per-source", type=int, default=0)
    parser.add_argument("--max-total-records", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/teacher_supervised/teacher_outputs.jsonl"))
    parser.add_argument("--progress", type=Path, default=Path("data/teacher_supervised/progress.json"))
    parser.add_argument("--target-estimated-tokens", type=int, default=0)
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()

    base_url = args.base_url
    if base_url is None:
        base_url = DEFAULT_OLLAMA_BASE_URL if args.runtime == "ollama" else DEFAULT_LMSTUDIO_BASE_URL
    cfg = TeacherConfig(
        runtime=args.runtime,
        model=args.model,
        base_url=base_url,
        max_tokens=args.max_tokens_per_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
    )
    if args.input_jsonl is not None:
        prompt_records = iter_jsonl(args.input_jsonl)
    else:
        source_config = json.loads(args.source_config.read_text(encoding="utf-8"))
        source_names = {item.strip() for item in args.sources.split(",") if item.strip()} or None
        prompt_records = iter_prompt_records(
            source_config,
            source_names=source_names,
            max_chars=args.max_prompt_chars,
            media_dir=args.media_dir,
            max_records_per_source=args.max_records_per_source,
            max_total_records=args.max_total_records,
        )
    count = write_teacher_jsonl(args.output, generate_records(cfg, prompt_records, args.target_estimated_tokens, args.progress))
    print(json.dumps({"records_written": count, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
