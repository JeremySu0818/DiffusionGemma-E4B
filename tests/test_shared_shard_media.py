from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json

import numpy as np
import pytest

from diffusiongemma_e4b.corruption import _write_shard, verify_shards
from diffusiongemma_e4b.shard_media import DiskBudget, MediaStore, load_media_row
from diffusiongemma_e4b.train import CorruptionShardDataset, corruption_data_fingerprint
from diffusiongemma_e4b.validate import _first_batch, validate_shards


def examples():
    rng = np.random.default_rng(7)
    media = dict(pixel_values=rng.normal(size=(3, 32, 32)).astype(np.float16),
                 image_position_ids=np.arange(16).reshape(8, 2),
                 input_features=rng.normal(size=(8, 32)).astype(np.float16),
                 input_features_mask=np.ones(8, dtype=np.int64))
    return [dict(prefix_ids=np.array([1, 2, 3]), attention_mask=np.ones(3, dtype=np.int64),
                 prefix_lens=np.array(3), target_ids=np.array([4, 5, -100, -100]),
                 record_id=f"record-{i // 2}", source_id="test", bucket="test",
                 chunk_index=i % 2, modality="multimodal",
                 mm_token_type_ids=np.array([0, 1, 0]), **media) for i in range(8)]


def test_shared_media_roundtrip_deduplicates_across_shards_and_concurrent_writes(tmp_path):
    budget = DiskBudget(tmp_path, 0)
    store = MediaStore(tmp_path, budget)
    rows = examples()
    with ThreadPoolExecutor(2) as pool:
        paths = list(pool.map(lambda i: _write_shard(tmp_path, "image", i, rows[i*4:(i+1)*4],
                                                   4, 3, vocab_size=32, media_store=store,
                                                   disk_budget=budget), range(2)))
    assert len(list((tmp_path / "media").glob("*.npz"))) == 4
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            assert "pixel_values" not in shard
            assert len(set(shard["pixel_values_refs"])) == 1
            for key in ("pixel_values", "image_position_ids", "input_features", "input_features_mask"):
                np.testing.assert_array_equal(load_media_row(tmp_path, shard, key, 0), rows[0][key])
    assert verify_shards(tmp_path, canvas_length=4, expected_blocks=8)["exact"]
    assert validate_shards(tmp_path, expected_canvas_length=4)["blocks"] == 8
    first = _first_batch(tmp_path)
    np.testing.assert_array_equal(first["pixel_values"][0].numpy(), rows[0]["pixel_values"])
    data = CorruptionShardDataset(tmp_path, val_fraction=.25)
    np.testing.assert_array_equal(data[0]["input_features"].numpy(), rows[0]["input_features"])
    # Sidecars are part of the training fingerprint, and corruption is detected.
    fingerprint = corruption_data_fingerprint(tmp_path)
    media = next((tmp_path / "media").glob("pixel_values-*.npz"))
    np.savez_compressed(media, value=np.zeros_like(rows[0]["pixel_values"]))
    assert corruption_data_fingerprint(tmp_path) != fingerprint
    with pytest.raises(ValueError, match="Corrupt shared media"):
        verify_shards(tmp_path, canvas_length=4)
    media.unlink()
    with pytest.raises(FileNotFoundError):
        validate_shards(tmp_path, expected_canvas_length=4)


def test_disk_budget_accounts_for_concurrent_writers_and_releases(monkeypatch, tmp_path):
    monkeypatch.setattr("diffusiongemma_e4b.shard_media.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=5 * 1024**2))
    budget = DiskBudget(tmp_path, 2 * 1024**2)
    with budget.claim(512 * 1024):
        with pytest.raises(OSError, match="Insufficient disk space"):
            with budget.claim(512 * 1024):
                pass
    assert budget.pending == 0
    with budget.claim(0):
        pass


def test_low_disk_space_does_not_publish_partial_media(monkeypatch, tmp_path):
    monkeypatch.setattr("diffusiongemma_e4b.shard_media.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=0))
    store = MediaStore(tmp_path, DiskBudget(tmp_path, 0))
    with pytest.raises(OSError, match="Insufficient disk space"):
        store.put("pixel_values", examples()[0]["pixel_values"])
    assert not list((tmp_path / "media").iterdir())


def test_multimodal_build_resumes_shared_media_without_duplication(monkeypatch, tmp_path):
    from diffusiongemma_e4b import corruption

    class Tokenizer:
        pad_token_id = 0
        unk_token_id = None
        eos_token_id = 9

        def __len__(self):
            return 32

        def encode(self, text, **kwargs):
            return [1] * len(text)

        def convert_tokens_to_ids(self, token):
            return 9

    monkeypatch.setattr(corruption.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(_commit_hash=None))
    monkeypatch.setattr(corruption.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(corruption.AutoProcessor, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(corruption, "_CHECKPOINT_INTERVAL_SECONDS", 0)
    sample = examples()[0]
    calls = 0

    def encoder(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt
        return {key: sample[key] for key in ("prefix_ids", "attention_mask", "prefix_lens",
                                             "pixel_values", "input_features", "input_features_mask",
                                             "image_position_ids", "mm_token_type_ids")}

    monkeypatch.setattr(corruption, "_encoder_inputs", encoder)
    raw = tmp_path / "teacher.jsonl"
    raw.write_text("".join(json.dumps(dict(id=f"r{i}", text="abcdefgh", modality="image")) + "\n"
                           for i in range(3)))
    output = tmp_path / "shards"
    options = dict(tokenizer_name_or_path="fake", target_blocks=0, canvas_length=4,
                   record_order="source", shard_blocks=1)
    with pytest.raises(KeyboardInterrupt):
        corruption.build_shards(raw, output, **options)
    checkpoint = next(tmp_path.glob(".shards.staging-*/corruption_checkpoint.json"))
    assert json.loads(checkpoint.read_text())["records_read"] == 1
    result = corruption.build_shards(raw, output, **options)
    assert result["records_read"] == 3
    assert result["blocks_written"] == 9
    assert result["media_storage"] == "content_addressed_npz"
    assert len(list((output / "media").glob("*.npz"))) == 4
    assert verify_shards(output, canvas_length=4, expected_blocks=9)["exact"]
