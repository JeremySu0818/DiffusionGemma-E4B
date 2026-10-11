from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import signal
import shutil
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from .constants import CANVAS_LENGTH, DEFAULT_DIFFUSION_REFERENCE, DEFAULT_PREFIX_LENGTH


MULTIMODAL_SHARD_KEYS = (
    "pixel_values",
    "input_features",
    "input_features_mask",
    "image_position_ids",
    "mm_token_type_ids",
)
MANIFEST_NAME = "corruption_manifest.json"
_CHECKPOINT_INTERVAL_SECONDS = 60


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
    # prefix_length is a padding floor, never a content budget.
    prefix_length = max(prefix_length, len(ids))
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
        "padding": False,
        "truncation": False,
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


def _target_token_ids(tokenizer, text: str, finish_reason: str | None = None) -> list[int]:
    ids = list(tokenizer.encode(text.strip(), add_special_tokens=False))
    if not ids:
        return []
    token_id = _special_token_id(tokenizer, "<end_of_turn>")
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", None)
    if finish_reason != "length" and token_id is not None and ids[-1] != int(token_id):
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
    # Group nearby lengths to bound padding without opening one buffer for
    # every distinct prompt length. Media patch/audio dimensions stay exact.
    width = 1 << (len(example["prefix_ids"]) - 1).bit_length()
    return (("prefix_ids", (width,), str(example["prefix_ids"].dtype)),) + tuple(
        (key, (width,) if key == "mm_token_type_ids" else tuple(example[key].shape), str(example[key].dtype))
        for key in MULTIMODAL_SHARD_KEYS
        if key in example
    )


def _signature_name(signature: tuple[tuple[str, tuple[int, ...], str], ...]) -> str:
    keys = {item[0] for item in signature}
    if keys == {"prefix_ids"}:
        return "text"
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
    pad_token_id: int = 0,
) -> Path:
    width = max(len(ex["prefix_ids"]) for ex in examples)

    def padded(key: str, value: int) -> np.ndarray:
        return np.stack([
            np.pad(ex[key], (width - len(ex[key]), 0), constant_values=value)
            for ex in examples
        ])

    arrays = {
        "prefix_ids": padded("prefix_ids", pad_token_id).astype(np.int64),
        "attention_mask": padded("attention_mask", 0).astype(np.int64),
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
        "prefix_length": np.array([width], dtype=np.int64),
    }
    for key in MULTIMODAL_SHARD_KEYS:
        if key in examples[0]:
            arrays[key] = padded(key, 0) if key == "mm_token_type_ids" else np.stack([ex[key] for ex in examples])
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


def iter_jsonl_records(path: Path, record_order: str, seed: int, start_record: int = 0) -> Iterable[dict[str, Any]]:
    if record_order == "source":
        with path.open("rb") as source:
            index = 0
            for line in source:
                if not line.strip():
                    continue
                if index >= start_record:
                    yield json.loads(line)
                index += 1
        return
    offsets = _jsonl_offsets(path)
    rng = np.random.default_rng(seed)
    rng.shuffle(offsets)
    with path.open("rb") as f:
        for offset in offsets[start_record:]:
            f.seek(offset)
            yield json.loads(f.readline().decode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_jsonl(path: Path) -> tuple[str, int]:
    """Count nonempty records while hashing the input, without parsing JSON."""
    digest = hashlib.sha256()
    records = 0
    with path.open("rb") as source, tqdm(
        total=path.stat().st_size, desc="count records", unit="B", unit_scale=True,
    ) as progress:
        for line in source:
            digest.update(line)
            records += bool(line.strip())
            progress.update(len(line))
    return digest.hexdigest(), records


def _build_fingerprint(raw_sha256: str, **kwargs: Any) -> str:
    payload = {"schema": 3, "raw_jsonl_sha256": raw_sha256, **kwargs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _example_multimodal_bytes(example: dict[str, Any]) -> int:
    return sum(int(example[key].nbytes) for key in MULTIMODAL_SHARD_KEYS if key in example)


def _example_bytes(example: dict[str, Any]) -> int:
    return sum(int(value.nbytes) for value in example.values() if isinstance(value, np.ndarray))


def _cpu_workers() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


_PREPROCESS_MODELS = None
_PREPROCESS_CANCEL = None


def _init_preprocessor(source, revision, inherited_models, cancellation):
    global _PREPROCESS_MODELS, _PREPROCESS_CANCEL
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _PREPROCESS_CANCEL = cancellation
    if inherited_models is not None:
        _PREPROCESS_MODELS = inherited_models
        return
    import torch
    torch.set_num_threads(int(os.environ.get("DG_CORRUPTION_NATIVE_THREADS", "1")))
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, revision=revision or None)
    try:
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, revision=revision or None)
    except (OSError, ValueError):
        processor = None
    _PREPROCESS_MODELS = tokenizer, processor


def _prepare_worker(task):
    record, prefix_length, canvas_length, max_blocks = task
    tokenizer, processor = _PREPROCESS_MODELS
    return _prepare_record(record, processor, tokenizer, prefix_length, canvas_length, max_blocks, _PREPROCESS_CANCEL)


def _prepare_record(record, processor, tokenizer, prefix_length, canvas_length, max_blocks=0, cancel_event=None):
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    ids = _target_token_ids(tokenizer, str(record.get("text") or ""),
                            finish_reason=metadata.get("teacher_finish_reason"))
    prepared = []
    prior_ids = []
    for start in range(0, len(ids), canvas_length):
        if cancel_event is not None and cancel_event.is_set():
            break
        if max_blocks > 0 and len(prepared) >= max_blocks:
            break
        chunk = ids[start:start + canvas_length]
        try:
            encoder = _encoder_inputs(processor, tokenizer, record, prefix_length, prior_ids=prior_ids)
        except Exception as exc:
            raise RuntimeError(f"failed to process encoder/media inputs for record {record.get('id')}") from exc
        prepared.append((chunk, encoder))
        prior_ids.extend(chunk)
    return record, prepared


def _ordered_prepare(records, prepare, executor, prefetch):
    """Bound outstanding results and consume in input order for deterministic RNG."""
    pending = deque()
    records = iter(records)
    exhausted = object()
    try:
        for _ in range(prefetch):
            record = next(records, exhausted)
            if record is exhausted:
                break
            pending.append(executor.submit(prepare, record))
        while pending:
            yield pending.popleft().result()
            record = next(records, exhausted)
            if record is not exhausted:
                pending.append(executor.submit(prepare, record))
    finally:
        for future in pending:
            future.cancel()


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
    max_multimodal_bytes: int = 0,
    max_shard_uncompressed_bytes: int = 2 * 1024**3,
    dataset_config: Path | None = None,
    source_names: set[str] | None = None,
    num_workers: int = 1,
    prefetch_records: int = 0,
) -> dict[str, Any]:
    if num_workers < 0 or prefetch_records < 0:
        raise ValueError("num_workers and prefetch_records must be nonnegative")
    num_workers = num_workers or _cpu_workers()
    prefetch_records = prefetch_records or num_workers
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
    total_records = None
    if target_blocks == 0:
        raw_sha256, total_records = _scan_jsonl(raw_jsonl)
    else:
        raw_sha256 = _sha256_file(raw_jsonl)
    dataset_config_sha256 = (
        _sha256_file(dataset_config)
        if dataset_config is not None and dataset_config.is_file()
        else None
    )
    configured_mix: dict[str, float] = {}
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
            verify_shards(output_dir, min_blocks=target_blocks, canvas_length=canvas_length,
                          expected_blocks=existing["blocks_written"], num_workers=num_workers)
            return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{fingerprint[:16]}"
    checkpoint_path = staging / "corruption_checkpoint.json"
    if staging.exists() and not checkpoint_path.is_file():
        legacy = staging.with_name(f"{staging.name}.legacy-{time.time_ns()}")
        os.replace(staging, legacy)
        tqdm.write(f"Old staging has no resumable checkpoint; preserved at {legacy}. Starting anew.")
    staging.mkdir(parents=True, exist_ok=True)
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
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("schema_version") != 1 or checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError("incompatible corruption checkpoint; use a new output directory")
        files = checkpoint["files"]
        for entry in files:
            path = staging / entry["name"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"checkpoint shard is missing or corrupt: {path}")
        committed = {entry["name"] for entry in files}
        # A crash can leave shards written after the last atomic checkpoint.
        for path in staging.glob("corruption_*.npz"):
            if path.name not in committed:
                path.unlink()
        blocks_written = checkpoint["blocks_written"]
        records_read = checkpoint["records_read"]
        multimodal_bytes = checkpoint["multimodal_uncompressed_bytes"]
        shard_ids = checkpoint["shard_families"]
        by_modality = checkpoint["by_modality"]
        by_bucket = checkpoint["by_bucket"]
        rng.bit_generator.state = checkpoint["rng_state"]
        tqdm.write(f"Resuming from {records_read:,} records / {blocks_written:,} blocks.")
    record_progress = target_blocks == 0
    pbar = tqdm(
        total=total_records if record_progress else target_blocks,
        desc="corrupt", unit="record" if record_progress else "block",
        initial=records_read if record_progress else blocks_written,
    )
    if record_progress:
        pbar.set_postfix(blocks=blocks_written, refresh=False)
    executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="shard-write") if num_workers > 1 else None
    pending_writes = deque()
    cancel_preparation = threading.Event()
    preprocess_executor = None
    if num_workers > 1:
        import torch
        torch.set_num_threads(int(os.environ.get("DG_CORRUPTION_NATIVE_THREADS", "1")))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        use_fork = "fork" in multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context("fork" if use_fork else "spawn")
        cancel_preparation = context.Event()
        preprocess_executor = ProcessPoolExecutor(
            max_workers=num_workers, mp_context=context, initializer=_init_preprocessor,
            initargs=(tokenizer_name_or_path, tokenizer_revision,
                      (tokenizer, processor) if use_fork else None, cancel_preparation),
        )
    tqdm.write(f"Shard CPU workers: {num_workers}; prefetch records: {prefetch_records}.")

    def prepare(record):
        if cancel_preparation.is_set():
            return record, []
        return _prepare_record(record, processor, tokenizer, prefix_length, canvas_length,
                               target_blocks, cancel_preparation)

    def write_progress() -> None:
        _atomic_json(
            progress_path,
            {
                "fingerprint": fingerprint,
                "records_read": records_read,
                "total_records": total_records,
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
        shard_ids[family] = shard_id + 1
        buffers[sig] = []
        buffer_bytes[sig] = 0
        def write():
            path = _write_shard(staging, family, shard_id, examples, canvas_length, prefix_length, pad_token_id)
            return {"name": path.name, "family": family, "rows": len(examples),
                    "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        if executor is None:
            files.append(write())
        else:
            pending_writes.append(executor.submit(write))
            if len(pending_writes) >= num_workers:
                files.append(pending_writes.popleft().result())
        write_progress()

    def save_checkpoint() -> None:
        # Commit only at record boundaries, with all buffers durable first.
        # If interrupted during flush/commit, the previous checkpoint still wins.
        for sig in list(buffers):
            flush(sig)
        while pending_writes:
            files.append(pending_writes.popleft().result())
        _atomic_json(checkpoint_path, {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "records_read": records_read,
            "blocks_written": blocks_written,
            "multimodal_uncompressed_bytes": multimodal_bytes,
            "shard_families": shard_ids,
            "files": files,
            "by_modality": by_modality,
            "by_bucket": by_bucket,
            "rng_state": rng.bit_generator.state,
        })
        write_progress()

    if not checkpoint_path.is_file():
        save_checkpoint()
    last_checkpoint = time.monotonic()
    stop_requested = False

    def request_stop(signum, frame) -> None:
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        tqdm.write("Stopping after the current record and saving a checkpoint; Ctrl+C again stops immediately.")

    previous_sigint = None
    if threading.current_thread() is threading.main_thread():
        previous_sigint = signal.signal(signal.SIGINT, request_stop)
    prepared_records = None
    try:
        records = (() if target_blocks > 0 and blocks_written >= target_blocks else
                   iter_jsonl_records(raw_jsonl, record_order=record_order, seed=seed, start_record=records_read))
        tasks = ((record, prefix_length, canvas_length, target_blocks) for record in records)
        prepared_records = (_ordered_prepare(tasks, _prepare_worker, preprocess_executor, prefetch_records)
                            if preprocess_executor is not None else map(prepare, records))
        for record, prepared in prepared_records:
            if target_blocks > 0 and blocks_written >= target_blocks:
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
            if not prepared:
                if record_progress:
                    pbar.update(1)
                if stop_requested or time.monotonic() - last_checkpoint >= _CHECKPOINT_INTERVAL_SECONDS:
                    save_checkpoint()
                    last_checkpoint = time.monotonic()
                if stop_requested:
                    raise KeyboardInterrupt
                continue
            for chunk_index, (chunk, encoder) in enumerate(prepared):
                if target_blocks > 0 and blocks_written >= target_blocks:
                    break
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
                if not record_progress:
                    pbar.update(1)
            if record_progress:
                pbar.set_postfix(blocks=blocks_written, refresh=False)
                pbar.update(1)
            if stop_requested or time.monotonic() - last_checkpoint >= _CHECKPOINT_INTERVAL_SECONDS:
                save_checkpoint()
                last_checkpoint = time.monotonic()
            if stop_requested:
                raise KeyboardInterrupt
            if target_blocks > 0 and blocks_written >= target_blocks:
                break

        pbar.set_description("corrupt (flush)")
        save_checkpoint()
        if stop_requested:
            raise KeyboardInterrupt
    except BaseException:
        tqdm.write("Interrupted; rerun the same command to resume from the last saved checkpoint.")
        raise
    finally:
        cancel_preparation.set()
        if prepared_records is not None and hasattr(prepared_records, "close"):
            prepared_records.close()
        if preprocess_executor is not None:
            preprocess_executor.shutdown(wait=True, cancel_futures=True)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        pbar.close()

    if target_blocks > 0 and blocks_written != target_blocks:
        raise RuntimeError(
            f"corruption dataset underfilled: wrote exactly {blocks_written} blocks, required {target_blocks}; "
            "generate more valid teacher tokens before training"
        )
    tqdm.write("Verifying completed shards...")
    verify_shards(staging, min_blocks=target_blocks, canvas_length=canvas_length,
                  expected_blocks=blocks_written, num_workers=num_workers)
    manifest: dict[str, Any] = {
        "schema_version": 3,
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
        "prefix_policy": "full_input_with_dynamic_padding",
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
            expected_blocks=blocks_written,
            num_workers=num_workers,
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
    num_workers: int = 1,
) -> dict[str, Any]:
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    num_workers = num_workers or _cpu_workers()
    manifest_path = output_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if manifest is not None:
        files = [output_dir / str(item["name"]) for item in manifest.get("files", [])]
    else:
        files = sorted(output_dir.glob("corruption_*.npz"))
    if not files:
        raise ValueError(f"no corruption shards found in {output_dir}")
    entries = {item["name"]: item for item in manifest["files"]} if manifest is not None else {}

    def verify_file(path):
        if not path.is_file():
            raise ValueError(f"manifest shard is missing: {path}")
        if manifest is not None:
            entry = entries[path.name]
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
            target_shape = shard["target_ids"].shape
            if target_shape[1] != canvas_length:
                raise ValueError(f"{path} has wrong canvas length")
            if shard["corrupted_ids"].shape != target_shape:
                raise ValueError(f"{path} corrupted/target shape mismatch")
            rows = target_shape[0]
            for key in ("corruption_masks", "noise_t", "prefix_ids"):
                if shard[key].shape[0] != rows:
                    raise ValueError(f"{path} {key} row count mismatch")
            if "pixel_values" in shard and shard["pixel_values"].shape[0] != rows:
                raise ValueError(f"{path} pixel_values row count mismatch")
            if "input_features" in shard and shard["input_features"].shape[0] != rows:
                raise ValueError(f"{path} input_features row count mismatch")
            return rows
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="shard-verify") as pool:
            total = sum(tqdm(pool.map(verify_file, files), total=len(files), desc="verify", unit="shard"))
    else:
        total = sum(map(verify_file, files))
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
    parser.add_argument("--target-blocks", type=int, default=0,
                        help="Explicit block budget; 0 processes every teacher answer block.")
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--prefix-length", type=int, default=DEFAULT_PREFIX_LENGTH,
                        help="Minimum padding width; full inputs and prior answer blocks are always retained.")
    parser.add_argument("--shard-blocks", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--record-order", choices=["shuffled", "source"], default="shuffled")
    parser.add_argument("--max-multimodal-bytes", type=int, default=0,
                        help="Deprecated compatibility option; cumulative media bytes no longer stop dataset creation.")
    parser.add_argument("--max-shard-uncompressed-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--sources", default="")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="CPU workers for preprocessing, compression and verification; 0 uses all available CPUs, 1 is serial.")
    parser.add_argument("--prefetch-records", type=int, default=0,
                        help="Maximum queued preprocessing records; 0 uses the worker count. Lower to reduce RAM use.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only and args.num_workers != 1:
        import torch
        torch.set_num_threads(int(os.environ.get("DG_CORRUPTION_NATIVE_THREADS", "1")))

    if args.verify_only:
        result = verify_shards(
            args.output_dir,
            min_blocks=args.target_blocks,
            canvas_length=args.canvas_length,
            expected_blocks=args.target_blocks or None,
            num_workers=args.num_workers,
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
            num_workers=args.num_workers,
            prefetch_records=args.prefetch_records,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
