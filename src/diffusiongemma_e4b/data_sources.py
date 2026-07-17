from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


class SourceMixError(RuntimeError):
    """Raised when the configured training mixture cannot be satisfied."""


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


def _load_dataset_iter(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
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
    ds = load_dataset(source["id"], config, **kwargs) if config else load_dataset(source["id"], **kwargs)
    return iter(ds)


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


def iter_prompt_records(
    config: dict[str, Any],
    source_names: set[str] | None,
    max_chars: int,
    media_dir: Path | None,
    max_records_per_source: int = 0,
    max_total_records: int = 0,
) -> Iterable[dict[str, Any]]:
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
    all_states: list[_SourceState] = []
    for source in sources:
        bucket = str(source.get("bucket") or ("image" if "image" in str(source.get("modality")) else "text"))
        try:
            rows = _load_manifest_iter(source) if source.get("manifest_only") else _load_dataset_iter(source)
            state = _SourceState(source, iter(rows), _bounded_limit(source, max_records_per_source))
            by_bucket[bucket].append(state)
            all_states.append(state)
        except Exception as exc:  # noqa: BLE001
            failures.append({"source": str(source.get("id")), "phase": "open", "error": repr(exc)})

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
                states.pop(cursor)
                if states:
                    source_cursor[bucket] %= len(states)
                continue
            try:
                row = next(state.rows)
                state.rows_read += 1
            except StopIteration:
                states.pop(cursor)
                if states:
                    source_cursor[bucket] %= len(states)
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append({"source": str(state.source.get("id")), "phase": "iterate", "error": repr(exc)})
                states.pop(cursor)
                if states:
                    source_cursor[bucket] %= len(states)
                continue
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
    )
    result = write_jsonl(args.output, rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
