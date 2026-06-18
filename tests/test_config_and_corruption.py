from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffusiongemma_e4b.data_contract import SelfContinuationRecord


def test_self_continuation_record_rejects_prompt():
    record = SelfContinuationRecord(
        id="x",
        source_model="m",
        runtime="lmstudio",
        text="generated",
        estimated_tokens=2,
        prompt_text="hand written",
    )
    try:
        record.validate_formal()
    except ValueError as exc:
        assert "prompt_text" in str(exc)
    else:
        raise AssertionError("expected prompt rejection")


def test_corruption_shard_shape(tmp_path: Path):
    path = tmp_path / "corruption_000000.npz"
    np.savez_compressed(
        path,
        prefix_ids=np.zeros((2, 8), dtype=np.uint32),
        prefix_lens=np.array([0, 8], dtype=np.uint16),
        target_ids=np.zeros((2, 256), dtype=np.uint32),
        corrupted_ids=np.ones((2, 256), dtype=np.uint32),
        corruption_masks=np.ones((2, 256), dtype=np.bool_),
        noise_t=np.array([0.1, 0.5], dtype=np.float32),
    )
    with np.load(path) as shard:
        assert shard["target_ids"].shape == (2, 256)
        assert shard["corrupted_ids"].shape == shard["target_ids"].shape
