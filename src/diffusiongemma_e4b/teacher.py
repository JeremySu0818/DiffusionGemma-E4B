from __future__ import annotations

import argparse
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
from .data_contract import SelfContinuationRecord, write_jsonl


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
    def generate(self) -> str:
        raise NotImplementedError


class OpenAICompletionsClient(TeacherClient):
    """Prompt-free completions against LM Studio or llama.cpp OpenAI-compatible APIs."""

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def generate(self) -> str:
        payload = {
            "model": self.cfg.model,
            "prompt": "",
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
            return self._generate_chat_skeleton()
        data = response.json()
        return data["choices"][0].get("text", "")

    def _generate_chat_skeleton(self) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": ""}],
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
                f"OpenAI-compatible prompt-free completion failed: {response.status_code} {response.text}",
                response=response,
            )
        data = response.json()
        return data["choices"][0]["message"].get("content", "")


class OllamaGenerateClient(TeacherClient):
    """Prompt-free Ollama generation. Uses empty prompt and raw mode."""

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def generate(self) -> str:
        payload = {
            "model": self.cfg.model,
            "prompt": "",
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


def generate_records(cfg: TeacherConfig, target_estimated_tokens: int, progress_path: Path) -> Iterable[SelfContinuationRecord]:
    client = make_client(cfg)
    state = read_progress(progress_path)
    while state["estimated_tokens"] < target_estimated_tokens:
        started = time.time()
        text = client.generate()
        tok = estimate_tokens(text)
        record = SelfContinuationRecord(
            id=str(uuid.uuid4()),
            source_model=cfg.model,
            runtime=cfg.runtime,
            text=text,
            estimated_tokens=tok,
        )
        state["records"] += 1
        state["estimated_tokens"] += tok
        state["last_record_id"] = record.id
        state["last_seconds"] = round(time.time() - started, 3)
        state["last_estimated_tokens"] = tok
        save_progress(progress_path, state)
        yield record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=["lmstudio", "ollama", "llamacpp", "openai-compatible"], default="lmstudio")
    parser.add_argument("--model", default="google/gemma-4-e4b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output", type=Path, default=Path("data/raw_self_continuation/self_continuation.jsonl"))
    parser.add_argument("--progress", type=Path, default=Path("data/raw_self_continuation/progress.json"))
    parser.add_argument("--target-estimated-tokens", type=int, default=50_000_000)
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
    count = write_jsonl(args.output, generate_records(cfg, args.target_estimated_tokens, args.progress))
    print(json.dumps({"records_written": count, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
