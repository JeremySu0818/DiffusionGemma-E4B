"""Remove identifiable input-truncated teacher records in place, without backups."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import tempfile

from tokenizers import Tokenizer


def _stored_prompt(record: dict) -> str:
    prompt = str(record.get("prompt_text") or record.get("prompt") or "").strip()
    context = str(record.get("context_text") or record.get("context") or "").strip()
    policy = record.get("metadata", {}).get("teacher_input_policy")
    # Before full_stored_prompt, teacher wrapped even identical context/prompt.
    if context and prompt and (policy != "full_stored_prompt" or context != prompt):
        return f"Context:\n{context}\n\nRequest:\n{prompt}"
    return context or prompt


def clean_teacher(
    path: Path,
    tokenizer: Tokenizer,
    *,
    drop_length_limited: bool = False,
) -> dict:
    """Compare saved input with teacher accounting; preserve retained JSONL bytes."""
    tokenizer.no_truncation()
    tokenizer.no_padding()
    counts: Counter = Counter()
    reasons: Counter = Counter()
    with path.open("rb") as source, tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as output:
        temporary_path = Path(output.name)
        lines = (line for line in source if line.strip())
        while batch := list(islice(lines, 256)):
            records = [json.loads(line) for line in batch]
            encodings = tokenizer.encode_batch(
                [_stored_prompt(record) for record in records], add_special_tokens=False,
            )
            for line, record, encoding in zip(batch, records, encodings):
                counts["records_scanned"] += 1
                metadata = record.get("metadata", {})
                input_tokens = metadata.get("teacher_input_tokens")
                finish_reason = metadata.get("teacher_finish_reason")
                record_reasons = []
                if input_tokens is None:
                    counts["missing_input_accounting"] += 1
                elif len(encoding.ids) > input_tokens:
                    record_reasons.append("input_truncated")
                elif len(encoding.ids) < input_tokens:
                    counts["input_accounting_exceeds_reconstructed_prompt"] += 1
                if finish_reason is None:
                    counts["unknown_finish_reason"] += 1
                elif finish_reason == "length":
                    counts["length_limited_records"] += 1
                    if drop_length_limited:
                        record_reasons.append("length_limited")
                tokens = record["estimated_tokens"]
                if record_reasons:
                    counts["records_removed"] += 1
                    counts["output_tokens_removed"] += tokens
                    reasons.update(record_reasons)
                else:
                    output.write(line)
                    counts["records_kept"] += 1
                    counts["output_tokens_kept"] += tokens
    os.replace(temporary_path, path)
    return {"path": str(path), "counts": dict(counts), "removal_reasons": dict(reasons)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-json", type=Path, required=True,
        help="The same tokenizer.json used for teacher_input_tokens accounting.",
    )
    parser.add_argument(
        "--drop-length-limited", action="store_true",
        help="Also remove known length-stopped answers; otherwise retain them as partial targets.",
    )
    args = parser.parse_args()
    report = clean_teacher(
        args.jsonl, Tokenizer.from_file(str(args.tokenizer_json)),
        drop_length_limited=args.drop_length_limited,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
