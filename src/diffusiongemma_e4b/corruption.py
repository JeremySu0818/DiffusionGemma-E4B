from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from .constants import CANVAS_LENGTH, DEFAULT_BASE_MODEL, DEFAULT_PREFIX_LENGTH
from .data_contract import iter_jsonl


def _tokenize_texts(tokenizer, records: Iterable[dict]) -> list[int]:
    all_ids: list[int] = []
    for record in records:
        text = record.get("text", "")
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
    return all_ids


def _block_slices(total_tokens: int, canvas_length: int) -> int:
    return total_tokens // canvas_length


def _noise_block(target: np.ndarray, vocab_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    t = float(rng.uniform(0.05, 0.95))
    mask = rng.random(target.shape, dtype=np.float32) < t
    corrupted = target.copy()
    corrupted[mask] = rng.integers(0, vocab_size, size=int(mask.sum()), dtype=np.uint32)
    return corrupted, mask.astype(np.bool_), t


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
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start_block = int(progress.get("blocks_written", 0))
    else:
        start_block = 0

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    token_ids = _tokenize_texts(tokenizer, iter_jsonl(raw_jsonl))
    total_blocks = min(_block_slices(len(token_ids), canvas_length), target_blocks)
    if total_blocks <= start_block:
        return {"blocks_written": start_block, "total_blocks_available": total_blocks, "done": True}

    ids = np.asarray(token_ids[: total_blocks * canvas_length], dtype=np.uint32)
    rng = np.random.default_rng(seed + start_block)
    block_index = start_block
    pbar = tqdm(total=total_blocks - start_block, desc="corrupt", unit="block")
    while block_index < total_blocks:
        n = min(shard_blocks, total_blocks - block_index)
        prefix_ids = np.zeros((n, prefix_length), dtype=np.uint32)
        prefix_lens = np.zeros((n,), dtype=np.uint16)
        target_ids = np.zeros((n, canvas_length), dtype=np.uint32)
        corrupted_ids = np.zeros((n, canvas_length), dtype=np.uint32)
        corruption_masks = np.zeros((n, canvas_length), dtype=np.bool_)
        noise_t = np.zeros((n,), dtype=np.float32)

        for row in range(n):
            b = block_index + row
            start = b * canvas_length
            target = ids[start : start + canvas_length]
            prefix_start = max(0, start - prefix_length)
            prefix = ids[prefix_start:start]
            prefix_lens[row] = len(prefix)
            if len(prefix):
                prefix_ids[row, -len(prefix) :] = prefix
            target_ids[row] = target
            corrupted, mask, t = _noise_block(target, tokenizer.vocab_size, rng)
            corrupted_ids[row] = corrupted
            corruption_masks[row] = mask
            noise_t[row] = t

        shard_id = block_index // shard_blocks
        shard_path = output_dir / f"corruption_{shard_id:06d}.npz"
        np.savez_compressed(
            shard_path,
            prefix_ids=prefix_ids,
            prefix_lens=prefix_lens,
            target_ids=target_ids,
            corrupted_ids=corrupted_ids,
            corruption_masks=corruption_masks,
            noise_t=noise_t,
            canvas_length=np.array([canvas_length], dtype=np.uint16),
            prefix_length=np.array([prefix_length], dtype=np.uint16),
        )
        block_index += n
        progress = {
            "blocks_written": block_index,
            "target_blocks": target_blocks,
            "total_blocks_available": total_blocks,
            "canvas_length": canvas_length,
            "prefix_length": prefix_length,
            "tokenizer": tokenizer_name_or_path,
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        pbar.update(n)
    pbar.close()
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
            total += shard["target_ids"].shape[0]
    if total < min_blocks:
        raise ValueError(f"only {total} blocks found; need {min_blocks}")
    return {"files": len(files), "blocks": total, "canvas_length": canvas_length}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=Path, default=Path("data/raw_self_continuation/self_continuation.jsonl"))
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
