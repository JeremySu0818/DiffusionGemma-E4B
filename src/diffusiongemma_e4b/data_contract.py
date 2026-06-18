from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .constants import FORBIDDEN_PROMPT_PATH_PARTS, FORBIDDEN_TEST_ONLY_PROMPTS


@dataclass
class SelfContinuationRecord:
    id: str
    source_model: str
    runtime: str
    text: str
    estimated_tokens: int
    prompt_source: str = "prompt_free"
    prompt_text: str = ""

    def validate_formal(self) -> None:
        if self.prompt_text.strip():
            raise ValueError("formal self-continuation record contains prompt_text")
        text_l = self.text.lower()
        for forbidden in FORBIDDEN_TEST_ONLY_PROMPTS:
            if forbidden.lower() in text_l:
                raise ValueError(f"record contains forbidden test-only prompt: {forbidden}")
        if self.prompt_source != "prompt_free":
            raise ValueError(f"invalid prompt_source for formal data: {self.prompt_source}")


def reject_curated_prompt_path(path: Path) -> None:
    parts = {p.lower().replace(".txt", "").replace(".md", "") for p in path.parts}
    bad = parts.intersection(FORBIDDEN_PROMPT_PATH_PARTS)
    if bad:
        raise ValueError(f"formal data cannot read curated prompt paths: {sorted(bad)}")


def write_jsonl(path: Path, records: Iterable[SelfContinuationRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            record.validate_formal()
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterable[dict]:
    reject_curated_prompt_path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
