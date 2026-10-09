from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from diffusiongemma_e4b import teacher


def _row(index: int, fingerprint: str = "unit") -> dict:
    return {
        "id": f"r{index}", "text": f"answer  {index}", "estimated_tokens": 10,
        "source_model": "model", "metadata": {
            "prompt_record_id": f"p{index}", "source_index": index + 1,
            "bucket": "code", "generation_fingerprint": fingerprint,
        },
    }


def _write(path: Path, rows: list[dict], mode: str = "w") -> None:
    with path.open(mode) as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def test_restart_reuses_index_and_only_decodes_appended_records(tmp_path, monkeypatch):
    path = tmp_path / "output.jsonl"
    _write(path, [_row(i) for i in range(100)])
    assert teacher.progress_from_output(path)["records"] == 100
    decoded_rows = []
    original = teacher.json.loads

    def loads(payload, *args, **kwargs):
        value = original(payload, *args, **kwargs)
        if isinstance(value, dict) and "text" in value:
            decoded_rows.append(value)
        return value

    monkeypatch.setattr(teacher.json, "loads", loads)
    assert teacher.progress_from_output(path)["estimated_tokens"] == 1000
    ids, hashes = teacher._existing_output_sets(path)
    assert len(ids) == len(hashes) == 100
    assert not decoded_rows
    _write(path, [_row(100)], mode="a")
    state = teacher.progress_from_output(path)
    assert state["records"] == state["source_index"] == 101
    assert state["records_by_bucket"] == {"code": 101}
    assert state["estimated_tokens_by_bucket"] == {"code": 1010}
    assert state["last_record_id"] == "r100"
    assert decoded_rows == [_row(100)]
    ids, hashes = teacher._existing_output_sets(path)
    assert "p100" in ids
    assert hashlib.sha256(b"answer 100").hexdigest() in hashes
    assert len(decoded_rows) == 1


@pytest.mark.parametrize("change", ["truncate", "replace", "rewrite", "rewrite_and_grow"])
def test_changed_output_rebuilds_index(tmp_path, change):
    path = tmp_path / "output.jsonl"
    _write(path, [_row(i) for i in range(10)])
    teacher.progress_from_output(path)
    rows = [_row(20)] if change in {"truncate", "replace"} else [_row(i + 10) for i in range(10)]
    if change == "replace":
        replacement = tmp_path / "replacement"
        _write(replacement, rows)
        replacement.replace(path)
    else:
        if change == "rewrite_and_grow":
            rows.append(_row(30))
        _write(path, rows)
    state = teacher.progress_from_output(path)
    assert state["records"] == len(rows)
    ids, _ = teacher._existing_output_sets(path)
    assert ids == {row["metadata"]["prompt_record_id"] for row in rows}


def test_append_with_mixed_fingerprint_is_still_rejected(tmp_path):
    path = tmp_path / "output.jsonl"
    _write(path, [_row(0)])
    teacher.progress_from_output(path)
    _write(path, [_row(1, "different")], mode="a")
    with pytest.raises(RuntimeError, match="multiple generation fingerprints"):
        teacher.progress_from_output(path)


@pytest.mark.parametrize("valid_tail", [False, True])
def test_large_unterminated_tail_repair(tmp_path, valid_tail):
    path = tmp_path / "output.jsonl"
    _write(path, [_row(0)])
    teacher.progress_from_output(path)
    row = _row(1)
    row["text"] = "x" * 150_000
    with path.open("a") as output:
        payload = json.dumps(row)
        output.write(payload if valid_tail else payload[:-2])
    assert teacher.progress_from_output(path)["records"] == (2 if valid_tail else 1)
    assert path.read_bytes().endswith(b"\n")


def test_corrupt_cache_falls_back_to_durable_output(tmp_path):
    path = tmp_path / "output.jsonl"
    _write(path, [_row(0)])
    path.with_name(path.name + ".resume.sqlite3").write_bytes(b"broken cache")
    assert teacher.progress_from_output(path)["records"] == 1
    assert teacher._existing_output_sets(path)[0] == {"p0"}
