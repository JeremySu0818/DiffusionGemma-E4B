from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from .constants import CANVAS_LENGTH, DEFAULT_BASE_MODEL, DEFAULT_PREFIX_LENGTH
from .data_contract import iter_jsonl


MULTIMODAL_SHARD_KEYS = (
    "pixel_values",
    "input_features",
    "input_features_mask",
    "image_position_ids",
    "mm_token_type_ids",
)


def _block_slices(total_tokens: int, canvas_length: int) -> int:
    return total_tokens // canvas_length


def _noise_block(target: np.ndarray, vocab_size: int, pad_token_id: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    t = float(rng.uniform(0.05, 0.95))
    label_mask = target >= 0
    mask = (rng.random(target.shape, dtype=np.float32) < t) & label_mask
    corrupted = np.full(target.shape, pad_token_id, dtype=np.int64)
    corrupted[label_mask] = target[label_mask]
    corrupted[mask] = rng.integers(0, vocab_size, size=int(mask.sum()), dtype=np.int64)
    return corrupted, mask.astype(np.bool_), t


def _prefix_text(record: dict[str, Any]) -> str:
    prompt = str(record.get("prompt_text") or record.get("prompt") or "").strip()
    context = str(record.get("context_text") or record.get("context") or "").strip()
    if context and prompt and context != prompt:
        return f"{context}\n\n{prompt}"
    return prompt or context


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

    return [Image.open(path).convert("RGB") for path in image_paths]


def _load_audio(media: dict[str, Any]) -> tuple[list[np.ndarray] | None, int | None]:
    audio_items = _media_values(media, "audio", "audios")
    if not audio_items:
        return None, None
    import soundfile as sf

    arrays: list[np.ndarray] = []
    sampling_rate = None
    for item in audio_items:
        path = item.get("path") if isinstance(item, dict) else item
        item_sampling_rate = item.get("sampling_rate") if isinstance(item, dict) else None
        array, sr = sf.read(path, dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)
        arrays.append(array)
        sampling_rate = int(item_sampling_rate or sr)
    return arrays, sampling_rate


def _pad_ids(ids: list[int], pad_token_id: int, prefix_length: int) -> tuple[np.ndarray, np.ndarray, int]:
    ids = ids[-prefix_length:]
    input_ids = np.full((prefix_length,), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((prefix_length,), dtype=np.int64)
    if ids:
        input_ids[-len(ids) :] = np.asarray(ids, dtype=np.int64)
        attention_mask[-len(ids) :] = 1
    return input_ids, attention_mask, len(ids)


def _encoder_inputs(processor, tokenizer, record: dict[str, Any], prefix_length: int) -> dict[str, np.ndarray]:
    text = _prefix_text(record)
    media = dict(record.get("media") or {})
    if not media:
        input_ids, attention_mask, prefix_len = _pad_ids(
            tokenizer.encode(text, add_special_tokens=False),
            tokenizer.pad_token_id or 0,
            prefix_length,
        )
        return {
            "prefix_ids": input_ids,
            "attention_mask": attention_mask,
            "prefix_lens": np.asarray(prefix_len, dtype=np.int64),
        }

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
        if array.shape[0] == 1:
            array = array[0]
        arrays[key] = array

    input_ids = arrays.pop("input_ids").astype(np.int64)
    attention_mask = arrays.pop("attention_mask", np.ones_like(input_ids)).astype(np.int64)
    if input_ids.shape[0] != prefix_length:
        padded_ids, padded_mask, _ = _pad_ids(input_ids.astype(int).tolist(), tokenizer.pad_token_id or 0, prefix_length)
        input_ids = padded_ids
        attention_mask = padded_mask

    out = {
        "prefix_ids": input_ids,
        "attention_mask": attention_mask,
        "prefix_lens": np.asarray(int(attention_mask.sum()), dtype=np.int64),
    }
    for key in MULTIMODAL_SHARD_KEYS:
        if key in arrays:
            out[key] = arrays[key]
    return out


def _target_blocks(tokenizer, text: str, canvas_length: int) -> Iterable[np.ndarray]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return
    for start in range(0, len(ids), canvas_length):
        chunk = ids[start : start + canvas_length]
        target = np.full((canvas_length,), -100, dtype=np.int64)
        target[: len(chunk)] = np.asarray(chunk, dtype=np.int64)
        yield target


def _signature(example: dict[str, np.ndarray]) -> tuple[tuple[str, tuple[int, ...], str], ...]:
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


def _write_shard(output_dir: Path, family: str, shard_id: int, examples: list[dict[str, np.ndarray]], canvas_length: int, prefix_length: int) -> Path:
    arrays = {
        "prefix_ids": np.stack([ex["prefix_ids"] for ex in examples]).astype(np.int64),
        "prefix_lens": np.asarray([int(ex["prefix_lens"]) for ex in examples], dtype=np.int64),
        "target_ids": np.stack([ex["target_ids"] for ex in examples]).astype(np.int64),
        "corrupted_ids": np.stack([ex["corrupted_ids"] for ex in examples]).astype(np.int64),
        "corruption_masks": np.stack([ex["corruption_masks"] for ex in examples]).astype(np.bool_),
        "noise_t": np.asarray([float(ex["noise_t"]) for ex in examples], dtype=np.float32),
        "canvas_length": np.array([canvas_length], dtype=np.uint16),
        "prefix_length": np.array([prefix_length], dtype=np.uint16),
    }
    for key in MULTIMODAL_SHARD_KEYS:
        if key in examples[0]:
            arrays[key] = np.stack([ex[key] for ex in examples])
    shard_path = output_dir / f"corruption_{family}_{shard_id:06d}.npz"
    np.savez_compressed(shard_path, **arrays)
    return shard_path


def build_shards(
    raw_jsonl: Path,
    output_dir: Path,
    tokenizer_name_or_path: str,
    target_blocks: int,
    canvas_length: int = CANVAS_LENGTH,
    prefix_length: int = DEFAULT_PREFIX_LENGTH,
    shard_blocks: int = 4096,
    seed: int = 1337,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "corruption_progress.json"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    rng = np.random.default_rng(seed)
    pad_token_id = tokenizer.pad_token_id or 0
    buffers: dict[tuple[tuple[str, tuple[int, ...], str], ...], list[dict[str, np.ndarray]]] = {}
    shard_ids: dict[str, int] = {}
    blocks_written = 0
    records_read = 0
    pbar = tqdm(total=target_blocks, desc="corrupt", unit="block")

    def flush(sig: tuple[tuple[str, tuple[int, ...], str], ...]) -> None:
        examples = buffers.get(sig, [])
        if not examples:
            return
        family = _signature_name(sig)
        shard_id = shard_ids.get(family, 0)
        _write_shard(output_dir, family, shard_id, examples, canvas_length, prefix_length)
        shard_ids[family] = shard_id + 1
        buffers[sig] = []

    for record in iter_jsonl(raw_jsonl):
        if blocks_written >= target_blocks:
            break
        records_read += 1
        try:
            encoder = _encoder_inputs(processor, tokenizer, record, prefix_length)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"failed to process encoder/media inputs for record {record.get('id', records_read)}") from exc

        for target in _target_blocks(tokenizer, str(record.get("text") or ""), canvas_length):
            if blocks_written >= target_blocks:
                break
            corrupted, mask, t = _noise_block(target, tokenizer.vocab_size, pad_token_id, rng)
            example = {
                **encoder,
                "target_ids": target,
                "corrupted_ids": corrupted,
                "corruption_masks": mask,
                "noise_t": np.asarray(t, dtype=np.float32),
            }
            sig = _signature(example)
            buffers.setdefault(sig, []).append(example)
            blocks_written += 1
            pbar.update(1)
            if len(buffers[sig]) >= shard_blocks:
                flush(sig)

    for sig in list(buffers):
        flush(sig)
    pbar.close()
    progress = {
        "records_read": records_read,
        "blocks_written": blocks_written,
        "target_blocks": target_blocks,
        "canvas_length": canvas_length,
        "prefix_length": prefix_length,
        "tokenizer": tokenizer_name_or_path,
        "shard_families": shard_ids,
    }
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    return progress


def verify_shards(output_dir: Path, min_blocks: int = 1, canvas_length: int = CANVAS_LENGTH) -> dict:
    files = sorted(output_dir.glob("corruption_*.npz"))
    total = 0
    for path in files:
        with np.load(path) as shard:
            if shard["target_ids"].shape[1] != canvas_length:
                raise ValueError(f"{path} has wrong canvas length")
            if shard["corrupted_ids"].shape != shard["target_ids"].shape:
                raise ValueError(f"{path} corrupted/target shape mismatch")
            if "pixel_values" in shard and shard["prefix_ids"].shape[0] != shard["pixel_values"].shape[0]:
                raise ValueError(f"{path} pixel_values row count mismatch")
            if "input_features" in shard and shard["prefix_ids"].shape[0] != shard["input_features"].shape[0]:
                raise ValueError(f"{path} input_features row count mismatch")
            total += shard["target_ids"].shape[0]
    if total < min_blocks:
        raise ValueError(f"only {total} blocks found; need {min_blocks}")
    return {"files": len(files), "blocks": total, "canvas_length": canvas_length}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=Path, default=Path("data/teacher_supervised/teacher_outputs.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/corruption"))
    parser.add_argument("--tokenizer", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--target-blocks", type=int, default=200_000)
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--prefix-length", type=int, default=DEFAULT_PREFIX_LENGTH)
    parser.add_argument("--shard-blocks", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        result = verify_shards(args.output_dir, min_blocks=args.target_blocks, canvas_length=args.canvas_length)
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
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
