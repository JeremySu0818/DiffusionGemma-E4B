from __future__ import annotations

import argparse
import io
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable


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
        role = _message_role(message)
        if role in {"user", "human", "prompter", "client"}:
            text = _message_text(message)
            if text:
                return text
    if messages:
        return _message_text(messages[0])
    return _first_text(row, ["prompt", "instruction", "question", "query", "input", "user", "title"])


def _extract_context(row: dict[str, Any]) -> str:
    context = _first_text(row, ["context", "document", "passage", "article", "text", "content", "body", "code"])
    if context:
        return context
    messages = _messages_from_row(row)
    user_turns = [_message_text(msg) for msg in messages if _message_role(msg) in {"user", "human", "prompter", "client"}]
    return "\n\n".join(turn for turn in user_turns[:3] if turn)


def _clean_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


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


def _collect_media(source: dict[str, Any], row: dict[str, Any], media_dir: Path | None) -> dict[str, Any]:
    media: dict[str, Any] = {}
    if isinstance(row.get("media"), dict):
        media.update(_jsonable(row["media"]))
    record_id = uuid.uuid4().hex
    source_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source.get("name") or source["id"]))

    def media_path(kind: str, index: int, suffix: str) -> Path:
        if media_dir is None:
            raise ValueError("media_dir is required to materialize decoded media objects")
        return media_dir / source_slug / f"{record_id}_{kind}_{index}{suffix}"

    for key in ("image", "images"):
        if key not in row:
            continue
        values = row[key] if isinstance(row[key], list) else [row[key]]
        paths = [_save_image(value, media_path("image", i, ".png")) for i, value in enumerate(values)]
        paths = [path for path in paths if path]
        if paths:
            media["images"] = paths

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
    prompt = _clean_text(_extract_user_prompt(row), max_chars)
    context = _clean_text(_extract_context(row), max_chars)

    if role.endswith("_warmup_or_context_seed") or role in {"plain_text_corpus", "code_context_stream", "math_science_context_stream"}:
        if not context:
            return None
        prompt = str(source.get("teacher_prompt") or "Respond helpfully using the provided context while preserving the teacher model's normal style.")

    if not prompt and not context:
        return None

    media = _collect_media(source, row, media_dir)

    return {
        "id": str(uuid.uuid4()),
        "source": source_id,
        "role": role,
        "modality": modality,
        "prompt_text": prompt,
        "context_text": context if context != prompt else "",
        "media": media,
        "metadata": {
            "dataset": source_id,
            "license_hint": str(source.get("license_hint") or ""),
            "target_policy": "teacher_generated_output_only",
        },
    }


def _load_dataset_iter(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Install the `datasets` package to prepare prompt/context banks.") from exc

    dataset_id = source["id"]
    config = source.get("config")
    split = source.get("split", "train")
    kwargs = {
        "split": split,
        "streaming": bool(source.get("streaming", True)),
        "trust_remote_code": bool(source.get("trust_remote_code", False)),
    }
    if config:
        ds = load_dataset(dataset_id, config, **kwargs)
    else:
        ds = load_dataset(dataset_id, **kwargs)
    return iter(ds)


def _load_manifest_iter(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    manifest_path = source.get("manifest_path")
    if manifest_path is None:
        name = source.get("name") or re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source["id"]))
        manifest_path = Path("data/manifests") / f"{name}.jsonl"
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{source['id']} is marked manifest_only, but manifest file is missing: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_prompt_records(
    config: dict[str, Any],
    source_names: set[str] | None,
    max_chars: int,
    media_dir: Path | None,
) -> Iterable[dict[str, Any]]:
    sources = [src for src in config["sources"] if src.get("enabled", True)]
    if source_names:
        sources = [src for src in sources if src["id"] in source_names or src.get("name") in source_names]
    active = []
    for source in sources:
        try:
            if source.get("manifest_only"):
                rows = _load_manifest_iter(source)
            else:
                rows = _load_dataset_iter(source)
            active.append((source, iter(rows)))
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps({"source_error": source.get("id"), "error": repr(exc)}, ensure_ascii=False),
                file=sys.stderr,
            )

    if not active:
        raise RuntimeError("No dataset sources could be opened.")
    while active:
        next_active = []
        for source, rows in active:
            try:
                row = next(rows)
            except StopIteration:
                continue
            except Exception as exc:  # noqa: BLE001
                print(
                    json.dumps({"source_error": source.get("id"), "error": repr(exc)}, ensure_ascii=False),
                    file=sys.stderr,
                )
                continue
            record = _prompt_record(source, row, max_chars=max_chars, media_dir=media_dir)
            if record is not None:
                yield record
            next_active.append((source, rows))
        active = next_active


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    by_source: dict[str, int] = {}
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    return {"records": count, "output": str(path), "by_source": by_source}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_sources.json"))
    parser.add_argument("--output", type=Path, default=Path("data/prompt_context/prompts.jsonl"))
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--media-dir", type=Path, default=Path("data/media_cache"))
    parser.add_argument("--sources", default="")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_names = {item.strip() for item in args.sources.split(",") if item.strip()} or None
    rows = iter_prompt_records(config, source_names=source_names, max_chars=args.max_chars, media_dir=args.media_dir)
    result = write_jsonl(args.output, rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
