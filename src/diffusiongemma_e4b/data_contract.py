from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TeacherSupervisedRecord:
    id: str
    source_model: str
    runtime: str
    prompt_text: str
    text: str
    estimated_tokens: int
    prompt_source: str
    modality: str = "text"
    context_text: str = ""
    media: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_formal(self) -> None:
        if not self.prompt_text.strip() and not self.context_text.strip() and not self.media:
            raise ValueError("teacher-supervised record requires prompt_text, context_text, or media")
        if self.prompt_source == "prompt_free":
            raise ValueError("prompt-free self-generation is disabled for the formal pipeline")
        if not self.text.strip():
            raise ValueError("teacher-supervised record has empty teacher output")


def write_teacher_jsonl(path: Path, records: Iterable[TeacherSupervisedRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            record.validate_formal()
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
