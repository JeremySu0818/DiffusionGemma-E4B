from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from .constants import CANVAS_LENGTH, DEFAULT_DIFFUSION_REFERENCE, DEFAULT_PREFIX_LENGTH
from .data_contract import iter_jsonl


MULTIMODAL_SHARD_KEYS = (
    "pixel_values",
    "input_features",
    "input_features_mask",
    "image_position_ids",
    "mm_token_type_ids",
)
MANIFEST_NAME = "corruption_manifest.json"


def _noise_block(
    target: np.ndarray,
    vocab_size: int,
    pad_token_id: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    t = float(rng.uniform(0.05, 0.95))
    label_mask = target >= 0
    mask = (rng.random(target.shape, dtype=np.float32) < t) & label_mask
    if label_mask.any() and not mask.any():
        valid = np.flatnonzero(label_mask)
        mask[int(rng.choice(valid))] = True
    corrupted = np.full(target.shape, pad_token_id, dtype=np.int64)
    corrupted[label_mask] = target[label_mask]
    corrupted[mask] = rng.integers(0, vocab_size, size=int(mask.sum()), dtype=np.int64)
    return corrupted, mask.astype(np.bool_), t


def _prefix_text(record: dict[str, Any]) -> str:
    prompt = str(record.get("prompt_text") or record.get("prompt") or "").strip()
    context = str(record.get("context_text") or record.get("context") or "").strip()
    if context and prompt and context != prompt:
        return f"Context:\n{context}\n\nRequest:\n{prompt}"
    return prompt or context


def _chat_prefix_text(
    tokenizer,
    record: dict[str, Any],
    prior_ids: list[int],
    image_count: int = 0,
    audio_count: int = 0,
    template_owner=None,
) -> str:
    user_text = _prefix_text(record)
    if not user_text:
        raise ValueError("teacher record has no prompt/context for encoder conditioning")
    content: str | list[dict[str, str]] = user_text
    if image_count or audio_count:
        content = (
            [{"type": "image"} for _ in range(image_count)]
            + [{"type": "audio"} for _ in range(audio_count)]
            + [{"type": "text", "text": user_text}]
        )
    owner = template_owner if template_owner is not None else tokenizer
    try:
        prefix = owner.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError):
        media = "<start_of_image>" * image_count + "<start_of_audio>" * audio_count
        prefix = f"<start_of_turn>user\n{media}{user_text}<end_of_turn>\n<start_of_turn>model\n"
    if prior_ids:
        prefix += tokenizer.decode(prior_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return prefix


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _media_values(media: dict[str, Any], *keys: str) -> list[Any]:
    values: list[Any] = []
    for key in keys:
        values.extend(_as_list(media.get(key)))
    return [value for value in values if value]


def _load_images(media: dict[str, Any]):
    image_paths = _media_values(media, "image", "images")
    if not image_paths:
        return None
    from PIL import Image

    images = []
    for value in image_paths:
        path = value.get("path") if isinstance(value, dict) else value
        images.append(Image.open(path).convert("RGB"))
    return images


def _load_audio(media: dict[str, Any]) -> tuple[list[np.ndarray] | None, int | None]:
    audio_items = _media_values(media, "audio", "audios")
    if not audio_items:
        return None, None
    import soundfile as sf

    arrays: list[np.ndarray] = []
    sampling_rate: int | None = None
    for item in audio_items:
        path = item.get("path") if isinstance(item, dict) else item
        declared_rate = item.get("sampling_rate") if isinstance(item, dict) else None
        array, file_rate = sf.read(path, dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)
        current_rate = int(declared_rate or file_rate)
        if sampling_rate is not None and current_rate != sampling_rate:
            raise ValueError("all audio items in one record must use the same sampling rate")
        arrays.append(array)
        sampling_rate = current_rate
    return arrays, sampling_rate


def _pad_ids(ids: list[int], pad_token_id: int, prefix_length: int) -> tuple[np.ndarray, np.ndarray, int]:
    ids = ids[-prefix_length:]
    input_ids = np.full((prefix_length,), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((prefix_length,), dtype=np.int64)
    if ids:
        input_ids[-len(ids) :] = np.asarray(ids, dtype=np.int64)
        attention_mask[-len(ids) :] = 1
    return input_ids, attention_mask, len(ids)


def _encoder_inputs(
    processor,
    tokenizer,
    record: dict[str, Any],
    prefix_length: int,
    prior_ids: list[int] | None = None,
) -> dict[str, np.ndarray]:
    media = dict(record.get("media") or {})
    image_values = _media_values(media, "image", "images")
    audio_values = _media_values(media, "audio", "audios")
    text = _chat_prefix_text(
        tokenizer,
        record,
        prior_ids or [],
        image_count=len(image_values),
        audio_count=len(audio_values),
        template_owner=processor if image_values or audio_values else None,
    )
    if not image_values and not audio_values:
        input_ids, attention_mask, prefix_len = _pad_ids(
            tokenizer.encode(text, add_special_tokens=False),
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
            prefix_length,
        )
        return {
            "prefix_ids": input_ids,
            "attention_mask": attention_mask,
            "prefix_lens": np.asarray(prefix_len, dtype=np.int64),
        }
    if processor is None:
        raise RuntimeError("AutoProcessor could not be loaded, but a multimodal teacher record was encountered.")

    kwargs: dict[str, Any] = {
        "text": text,
        "return_tensors": "np",
        "padding": "max_length",
        "truncation": True,
        "max_length": prefix_length,
    }
    images = _load_images(media)
    audio, sampling_rate = _load_audio(media)
    if images:
        kwargs["images"] = images[0] if len(images) == 1 else images
    if audio:
        kwargs["audio"] = audio[0] if len(audio) == 1 else audio
        kwargs["sampling_rate"] = sampling_rate
    encoded = processor(**kwargs)
    arrays: dict[str, np.ndarray] = {}
    for key, value in encoded.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        if key in {"pixel_values", "input_features"} and np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float16)
        arrays[key] = array

    input_ids = arrays.pop("input_ids").astype(np.int64)
    attention_mask = arrays.pop("attention_mask", np.ones_like(input_ids)).astype(np.int64)
    if input_ids.shape[0] != prefix_length:
        input_ids, attention_mask, _ = _pad_ids(
            input_ids.astype(int).tolist(),
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
            prefix_length,
        )
    out = {
        "prefix_ids": input_ids,
        "attention_mask": attention_mask,
        "prefix_lens": np.asarray(int(attention_mask.sum()), dtype=np.int64),
    }
    for key in MULTIMODAL_SHARD_KEYS:
        if key in arrays:
            out[key] = arrays[key]
    return out


def _special_token_id(tokenizer, token: str) -> int | None:
    try:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
    except (AttributeError, TypeError, ValueError):
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    return None if token_id < 0 or (unk is not None and token_id == int(unk)) else token_id


def _target_token_ids(tokenizer, text: str) -> list[int]:
    ids = list(tokenizer.encode(text.strip(), add_special_tokens=False))
    if not ids:
        return []
    token_id = _special_token_id(tokenizer, "<end_of_turn>")
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is not None and ids[-1] != int(token_id):
        ids.append(int(token_id))
    return ids


def _target_blocks(tokenizer, text: str, canvas_length: int) -> Iterable[np.ndarray]:
    ids = _target_token_ids(tokenizer, text)
    for start in range(0, len(ids), canvas_length):
        chunk = ids[start : start + canvas_length]
        target = np.full((canvas_length,), -100, dtype=np.int64)
        target[: len(chunk)] = np.asarray(chunk, dtype=np.int64)
        yield target


def _signature(example: dict[str, Any]) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (key, tuple(example[key].shape), str(example[key].dtype))
        for key in MULTIMODAL_SHARD_KEYS
        if key in example
    )


def _signature_name(signature: tuple[tuple[str, tuple[int, ...], str], ...]) -> str:
    if not signature:
        return "text"
    keys = {item[0] for item in signature}
    if "input_features" in keys:
        return "audio"
    if "pixel_values" in keys:
        return "image"
    return "multimodal"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _write_shard(
    output_dir: Path,
    family: str,
    shard_id: int,
    examples: list[dict[str, Any]],
    canvas_length: int,
    prefix_length: int,
) -> Path:
    arrays = {
        "prefix_ids": np.stack([ex["prefix_ids"] for ex in examples]).astype(np.int64),
        "attention_mask": np.stack([ex["attention_mask"] for ex in examples]).astype(np.int64),
        "prefix_lens": np.asarray([int(ex["prefix_lens"]) for ex in examples], dtype=np.int64),
        "target_ids": np.stack([ex["target_ids"] for ex in examples]).astype(np.int64),
        "corrupted_ids": np.stack([ex["corrupted_ids"] for ex in examples]).astype(np.int64),
        "corruption_masks": np.stack([ex["corruption_masks"] for ex in examples]).astype(np.bool_),
        "noise_t": np.asarray([float(ex["noise_t"]) for ex in examples], dtype=np.float32),
        "record_ids": np.asarray([str(ex["record_id"]) for ex in examples], dtype=np.str_),
        "source_ids": np.asarray([str(ex["source_id"]) for ex in examples], dtype=np.str_),
        "buckets": np.asarray([str(ex["bucket"]) for ex in examples], dtype=np.str_),
        "chunk_index": np.asarray([int(ex["chunk_index"]) for ex in examples], dtype=np.int32),
        "modality": np.asarray([str(ex["modality"]) for ex in examples], dtype=np.str_),
        "canvas_length": np.array([canvas_length], dtype=np.uint16),
        "prefix_length": np.array([prefix_length], dtype=np.uint16),
    }
    for key in MULTIMODAL_SHARD_KEYS:
        if key in examples[0]:
            arrays[key] = np.stack([ex[key] for ex in examples])
    shard_path = output_dir / f"corruption_{family}_{shard_id:06d}.npz"
    fd, temp_name = tempfile.mkstemp(prefix=f".{shard_path.stem}.", suffix=".npz", dir=output_dir)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        with Path(temp_name).open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temp_name, shard_path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return shard_path


def _jsonl_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def iter_jsonl_records(path: Path, record_order: str, seed: int) -> Iterable[dict[str, Any]]:
    if record_order == "source":
        yield from iter_jsonl(path)
        return
    offsets = _jsonl_offsets(path)
    rng = np.random.default_rng(seed)
    rng.shuffle(offsets)
    with path.open("rb") as f:
        for offset in offsets:
            f.seek(offset)
            yield json.loads(f.readline().decode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_fingerprint(raw_sha256: str, **kwargs: Any) -> str:
    payload = {"schema": 2, "raw_jsonl_sha256": raw_sha256, **kwargs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _example_multimodal_bytes(example: dict[str, Any]) -> int:
    return sum(int(example[key].nbytes) for key in MULTIMODAL_SHARD_KEYS if key in example)


def _example_bytes(example: dict[str, Any]) -> int:
    return sum(int(value.nbytes) for value in example.values() if isinstance(value, np.ndarray))


def _publish_staging(staging: Path, output_dir: Path, fingerprint: str) -> Path | None:
    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.parent / f".{output_dir.name}.stale-{fingerprint[:12]}-{int(time.time())}"
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        if backup is not None and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    return backup


def build_shards(
    raw_jsonl: Path,
    output_dir: Path,
    tokenizer_name_or_path: str,
    target_blocks: int,
    canvas_length: int = CANVAS_LENGTH,
    prefix_length: int = DEFAULT_PREFIX_LENGTH,
    shard_blocks: int = 4096,
    seed: int = 1337,
    record_order: str = "shuffled",
    max_multimodal_bytes: int = 64 * 1024**3,
    max_shard_uncompressed_bytes: int = 2 * 1024**3,
    dataset_config: Path | None = None,
    source_names: set[str] | None = None,
) -> dict[str, Any]:
    if target_blocks <= 0:
        raise ValueError("target_blocks must be positive for exact dataset verification")
    if not raw_jsonl.is_file():
        raise FileNotFoundError(raw_jsonl)
    tokenizer_config = AutoConfig.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=True
    )
    tokenizer_revision = str(
        getattr(tokenizer_config, "_commit_hash", None) or ""
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
        revision=tokenizer_revision or None,
    )
    raw_sha256 = _sha256_file(raw_jsonl)
    dataset_config_sha256 = (
        _sha256_file(dataset_config)
        if dataset_config is not None and dataset_config.is_file()
        else None
    )
    configured_mix: dict[str, float] = {}
    required_buckets: set[str] = set()
    if dataset_config is not None:
        if not dataset_config.is_file():
            raise FileNotFoundError(dataset_config)
        dataset_payload = json.loads(dataset_config.read_text(encoding="utf-8"))
        if source_names:
            selected_buckets = {
                str(item.get("bucket") or "")
                for item in dataset_payload.get("sources", [])
                if item.get("id") in source_names or item.get("name") in source_names
            }
        else:
            selected_buckets = {
                str(item.get("bucket") or "")
                for item in dataset_payload.get("sources", [])
                if item.get("enabled", True)
            }
        configured_mix = {
            str(item["bucket"]): float(item["share"])
            for item in dataset_payload.get("recommended_mix", [])
            if float(item.get("share") or 0.0) > 0
            and str(item["bucket"]) in selected_buckets
        }
        required_buckets = {
            str(item["bucket"])
            for item in dataset_payload.get("recommended_mix", [])
            if item.get("required", True)
            and float(item.get("share") or 0.0) > 0
            and str(item["bucket"]) in selected_buckets
        }
    fingerprint = _build_fingerprint(
        raw_sha256,
        tokenizer=tokenizer_name_or_path,
        tokenizer_revision=tokenizer_revision,
        target_blocks=target_blocks,
        canvas_length=canvas_length,
        prefix_length=prefix_length,
        shard_blocks=shard_blocks,
        seed=seed,
        record_order=record_order,
        max_multimodal_bytes=max_multimodal_bytes,
        max_shard_uncompressed_bytes=max_shard_uncompressed_bytes,
        dataset_config_sha256=dataset_config_sha256,
        source_names=sorted(source_names or []),
    )
    existing_manifest = output_dir / MANIFEST_NAME
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            verify_shards(output_dir, min_blocks=target_blocks, canvas_length=canvas_length, expected_blocks=target_blocks)
            return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{fingerprint[:16]}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    progress_path = staging / "corruption_progress.json"

    try:
        processor = AutoProcessor.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
            revision=tokenizer_revision or None,
        )
    except (OSError, ValueError):
        processor = None
    rng = np.random.default_rng(seed)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size
    buffers: dict[tuple[tuple[str, tuple[int, ...], str], ...], list[dict[str, Any]]] = {}
    buffer_bytes: dict[tuple[tuple[str, tuple[int, ...], str], ...], int] = {}
    shard_ids: dict[str, int] = {}
    files: list[dict[str, Any]] = []
    blocks_written = 0
    records_read = 0
    multimodal_bytes = 0
    by_modality: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    pbar = tqdm(total=target_blocks, desc="corrupt", unit="block")

    def write_progress() -> None:
        _atomic_json(
            progress_path,
            {
                "fingerprint": fingerprint,
                "records_read": records_read,
                "blocks_written": blocks_written,
                "target_blocks": target_blocks,
                "multimodal_uncompressed_bytes": multimodal_bytes,
                "shard_families": shard_ids,
            },
        )

    def flush(sig: tuple[tuple[str, tuple[int, ...], str], ...]) -> None:
        examples = buffers.get(sig, [])
        if not examples:
            return
        family = _signature_name(sig)
        shard_id = shard_ids.get(family, 0)
        path = _write_shard(staging, family, shard_id, examples, canvas_length, prefix_length)
        shard_ids[family] = shard_id + 1
        files.append(
            {
                "name": path.name,
                "family": family,
                "rows": len(examples),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        buffers[sig] = []
        buffer_bytes[sig] = 0
        write_progress()

    try:
        for record in iter_jsonl_records(raw_jsonl, record_order=record_order, seed=seed):
            if blocks_written >= target_blocks:
                break
            records_read += 1
            record_id = str(record.get("id") or f"row-{records_read}")
            source_id = str(record.get("prompt_source") or record.get("source") or "unknown")
            metadata = (
                record.get("metadata")
                if isinstance(record.get("metadata"), dict)
                else {}
            )
            bucket = str(metadata.get("bucket") or "unknown")
            modality = str(record.get("modality") or "text")
            target_ids = _target_token_ids(tokenizer, str(record.get("text") or ""))
            if not target_ids:
                continue
            prior_ids: list[int] = []
            for chunk_index, start in enumerate(range(0, len(target_ids), canvas_length)):
                if blocks_written >= target_blocks:
                    break
                chunk = target_ids[start : start + canvas_length]
                try:
                    encoder = _encoder_inputs(processor, tokenizer, record, prefix_length, prior_ids=prior_ids)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"failed to process encoder/media inputs for record {record_id}") from exc
                target = np.full((canvas_length,), -100, dtype=np.int64)
                target[: len(chunk)] = np.asarray(chunk, dtype=np.int64)
                corrupted, mask, t = _noise_block(target, vocab_size, pad_token_id, rng)
                example: dict[str, Any] = {
                    **encoder,
                    "target_ids": target,
                    "corrupted_ids": corrupted,
                    "corruption_masks": mask,
                    "noise_t": np.asarray(t, dtype=np.float32),
                    "record_id": record_id,
                    "source_id": source_id,
                    "bucket": bucket,
                    "chunk_index": chunk_index,
                    "modality": modality,
                }
                example_mm_bytes = _example_multimodal_bytes(example)
                multimodal_bytes += example_mm_bytes
                if max_multimodal_bytes > 0 and multimodal_bytes > max_multimodal_bytes:
                    raise RuntimeError(
                        "multimodal tensor preflight limit exceeded: "
                        f"{multimodal_bytes} > {max_multimodal_bytes} uncompressed bytes. "
                        "Reduce the image share/resolution/target blocks or raise DG_MAX_MULTIMODAL_SHARD_BYTES "
                        "only after confirming cloud disk and host-RAM capacity."
                    )
                sig = _signature(example)
                size = _example_bytes(example)
                if buffers.get(sig) and (
                    len(buffers[sig]) >= shard_blocks
                    or buffer_bytes.get(sig, 0) + size > max_shard_uncompressed_bytes
                ):
                    flush(sig)
                buffers.setdefault(sig, []).append(example)
                buffer_bytes[sig] = buffer_bytes.get(sig, 0) + size
                blocks_written += 1
                by_modality[modality] = by_modality.get(modality, 0) + 1
                by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
                pbar.update(1)
                prior_ids.extend(chunk)

        for sig in list(buffers):
            flush(sig)
    except BaseException:
        write_progress()
        raise
    finally:
        pbar.close()

    if blocks_written != target_blocks:
        raise RuntimeError(
            f"corruption dataset underfilled: wrote exactly {blocks_written} blocks, required {target_blocks}; "
            "generate more valid teacher tokens before training"
        )
    realized_shares = {
        bucket: count / blocks_written
        for bucket, count in sorted(by_bucket.items())
    }
    underfilled_buckets = {
        bucket: {
            "configured_prompt_share": configured_mix[bucket],
            "realized_block_share": realized_shares.get(bucket, 0.0),
        }
        for bucket in required_buckets
        if realized_shares.get(bucket, 0.0) < configured_mix[bucket] * 0.5
    }
    if underfilled_buckets:
        raise RuntimeError(
            "realized training-block mixture lost more than half of a required "
            f"bucket allocation: {underfilled_buckets}"
        )
    verify_shards(staging, min_blocks=target_blocks, canvas_length=canvas_length, expected_blocks=target_blocks)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "raw_jsonl": str(raw_jsonl),
        "raw_jsonl_sha256": raw_sha256,
        "tokenizer": tokenizer_name_or_path,
        "tokenizer_revision": tokenizer_revision,
        "target_blocks": target_blocks,
        "blocks_written": blocks_written,
        "records_read": records_read,
        "canvas_length": canvas_length,
        "prefix_length": prefix_length,
        "record_order": record_order,
        "seed": seed,
        "by_modality": by_modality,
        "by_bucket": by_bucket,
        "bucket_block_shares": {
            bucket: count / blocks_written
            for bucket, count in sorted(by_bucket.items())
        },
        "configured_prompt_mix": configured_mix,
        "dataset_config": str(dataset_config) if dataset_config is not None else None,
        "dataset_config_sha256": dataset_config_sha256,
        "multimodal_uncompressed_bytes": multimodal_bytes,
        "files": files,
    }
    _atomic_json(staging / MANIFEST_NAME, manifest)
    backup = _publish_staging(staging, output_dir, fingerprint)
    try:
        verified = verify_shards(
            output_dir,
            min_blocks=target_blocks,
            canvas_length=canvas_length,
            expected_blocks=target_blocks,
        )
    except BaseException:
        failed = output_dir.parent / f".{output_dir.name}.failed-{fingerprint[:12]}"
        if output_dir.exists():
            os.replace(output_dir, failed)
        if backup is not None and backup.exists():
            os.replace(backup, output_dir)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)
    manifest["verified"] = verified
    return manifest


def verify_shards(
    output_dir: Path,
    min_blocks: int = 1,
    canvas_length: int = CANVAS_LENGTH,
    expected_blocks: int | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if manifest is not None:
        files = [output_dir / str(item["name"]) for item in manifest.get("files", [])]
    else:
        files = sorted(output_dir.glob("corruption_*.npz"))
    if not files:
        raise ValueError(f"no corruption shards found in {output_dir}")
    total = 0
    for path in files:
        if not path.is_file():
            raise ValueError(f"manifest shard is missing: {path}")
        if manifest is not None:
            entry = next(item for item in manifest["files"] if item["name"] == path.name)
            if int(entry.get("bytes", -1)) != path.stat().st_size:
                raise ValueError(f"{path} size does not match manifest")
            if entry.get("sha256") != _sha256_file(path):
                raise ValueError(f"{path} sha256 does not match manifest")
        with np.load(path, allow_pickle=False) as shard:
            required = {
                "prefix_ids",
                "target_ids",
                "corrupted_ids",
                "corruption_masks",
                "noise_t",
                "record_ids",
                "source_ids",
                "buckets",
                "chunk_index",
                "modality",
            }
            missing = required - set(shard.files)
            if manifest is not None and missing:
                raise ValueError(f"{path} missing required arrays: {sorted(missing)}")
            if shard["target_ids"].shape[1] != canvas_length:
                raise ValueError(f"{path} has wrong canvas length")
            if shard["corrupted_ids"].shape != shard["target_ids"].shape:
                raise ValueError(f"{path} corrupted/target shape mismatch")
            rows = shard["target_ids"].shape[0]
            for key in ("corruption_masks", "noise_t", "prefix_ids"):
                if shard[key].shape[0] != rows:
                    raise ValueError(f"{path} {key} row count mismatch")
            if "pixel_values" in shard and shard["pixel_values"].shape[0] != rows:
                raise ValueError(f"{path} pixel_values row count mismatch")
            if "input_features" in shard and shard["input_features"].shape[0] != rows:
                raise ValueError(f"{path} input_features row count mismatch")
            total += rows
    if total < min_blocks:
        raise ValueError(f"only {total} blocks found; need at least {min_blocks}")
    if expected_blocks is not None and total != expected_blocks:
        raise ValueError(f"found {total} blocks; expected exactly {expected_blocks}")
    return {"files": len(files), "blocks": total, "canvas_length": canvas_length, "exact": expected_blocks is None or total == expected_blocks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=Path, default=Path("data/teacher_supervised/teacher_outputs.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/corruption"))
    parser.add_argument("--tokenizer", default=os.environ.get("DG_STUDENT_MODEL", DEFAULT_DIFFUSION_REFERENCE))
    parser.add_argument("--target-blocks", type=int, default=200_000)
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--prefix-length", type=int, default=DEFAULT_PREFIX_LENGTH)
    parser.add_argument("--shard-blocks", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--record-order", choices=["shuffled", "source"], default="shuffled")
    parser.add_argument("--max-multimodal-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--max-shard-uncompressed-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--sources", default="")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        result = verify_shards(
            args.output_dir,
            min_blocks=args.target_blocks,
            canvas_length=args.canvas_length,
            expected_blocks=args.target_blocks,
        )
    else:
        result = build_shards(
            args.raw_jsonl,
            args.output_dir,
            args.tokenizer,
            args.target_blocks,
            args.canvas_length,
            args.prefix_length,
            args.shard_blocks,
            args.seed,
            args.record_order,
            args.max_multimodal_bytes,
            args.max_shard_uncompressed_bytes,
            args.dataset_config,
            {item.strip() for item in args.sources.split(",") if item.strip()} or None,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
