from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from diffusiongemma_e4b.data_contract import TeacherSupervisedRecord, write_teacher_jsonl
from diffusiongemma_e4b.data_sources import SourceMixError, _prompt_record, iter_prompt_records
from diffusiongemma_e4b.corruption import (
    _chat_prefix_text,
    _target_blocks,
    build_shards,
    iter_jsonl_records,
    verify_shards,
)
from diffusiongemma_e4b.teacher import (
    OpenAICompletionsClient,
    TeacherConfig,
    _chat_content,
    read_progress,
)


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


def test_teacher_supervised_record_rejects_prompt_free(tmp_path: Path):
    record = TeacherSupervisedRecord(
        id="x",
        source_model="teacher",
        runtime="lmstudio",
        prompt_text="Explain diffusion denoising.",
        text="answer",
        estimated_tokens=2,
        prompt_source="prompt_free",
    )
    try:
        write_teacher_jsonl(tmp_path / "teacher.jsonl", [record])
    except ValueError as exc:
        assert "prompt-free" in str(exc)
    else:
        raise AssertionError("expected prompt-free rejection")


def test_prompt_record_extracts_user_message_without_answer_target():
    source = {
        "id": "unit/source",
        "role": "prompt_context_bank",
        "modality": "text",
        "license_hint": "unit",
    }
    row = {
        "messages": [
            {"role": "user", "content": "What is block diffusion?"},
            {"role": "assistant", "content": "Do not use this as target."},
        ]
    }
    record = _prompt_record(source, row, max_chars=1000)
    assert record is not None
    assert record["prompt_text"] == "What is block diffusion?"
    assert "Do not use" not in record["prompt_text"]
    assert record["metadata"]["target_policy"] == "teacher_generated_output_only"


def test_teacher_chat_content_includes_image_media(tmp_path: Path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
        b"\x00\x05\xfe\x02\xfeA\x0b\x83\xb1\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    content = _chat_content("describe this", {"images": [str(image_path)]})

    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_iter_prompt_records_honors_source_and_total_limits(tmp_path: Path):
    manifest_a = tmp_path / "a.jsonl"
    manifest_b = tmp_path / "b.jsonl"
    manifest_a.write_text(
        "\n".join(json.dumps({"prompt": prompt}) for prompt in ("a1", "a2", "a3")) + "\n",
        encoding="utf-8",
    )
    manifest_b.write_text(
        "\n".join(json.dumps({"prompt": prompt}) for prompt in ("b1", "b2", "b3")) + "\n",
        encoding="utf-8",
    )

    config = {
        "sources": [
            {"id": "a", "name": "a", "role": "prompt_context_bank", "modality": "text", "manifest_only": True, "manifest_path": str(manifest_a), "max_records": 2},
            {"id": "b", "name": "b", "role": "prompt_context_bank", "modality": "text", "manifest_only": True, "manifest_path": str(manifest_b), "max_records": 2},
        ]
    }

    rows = list(iter_prompt_records(config, source_names=None, max_chars=100, media_dir=tmp_path, max_total_records=3))

    assert len(rows) == 3
    assert [row["prompt_text"] for row in rows] == ["a1", "b1", "a2"]


def test_corruption_jsonl_shuffled_order_is_deterministic_and_not_source_order(tmp_path: Path):
    path = tmp_path / "teacher.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": str(i), "text": f"record {i}"}) for i in range(12)) + "\n",
        encoding="utf-8",
    )

    source_ids = [row["id"] for row in iter_jsonl_records(path, record_order="source", seed=7)]
    shuffled_ids = [row["id"] for row in iter_jsonl_records(path, record_order="shuffled", seed=7)]
    shuffled_ids_again = [row["id"] for row in iter_jsonl_records(path, record_order="shuffled", seed=7)]

    assert source_ids == [str(i) for i in range(12)]
    assert shuffled_ids == shuffled_ids_again
    assert shuffled_ids != source_ids


def test_teacher_progress_reconciles_from_output_jsonl(tmp_path: Path):
    output = tmp_path / "teacher_outputs.jsonl"
    output.write_text(
        "\n".join(
            json.dumps({"id": f"r{i}", "estimated_tokens": tokens})
            for i, tokens in enumerate([5, 7], start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"records": 3, "estimated_tokens": 99, "source_index": 3}), encoding="utf-8")

    state = read_progress(progress, output_path=output)

    assert state["records"] == 2
    assert state["estimated_tokens"] == 12
    assert state["source_index"] == 2
    assert state["last_record_id"] == "r2"
    assert state["reconciled_from_output"] is True


def test_production_dataset_profile_has_exact_training_mix_and_no_active_audio():
    config = json.loads(Path("configs/dataset_sources.json").read_text(encoding="utf-8"))

    assert sum(item["share"] for item in config["recommended_mix"]) == pytest.approx(1.0)
    assert all(item["bucket"] != "eval_only" for item in config["recommended_mix"])
    assert not any(
        source.get("enabled", True) and source.get("modality") == "audio"
        for source in config["sources"]
    )
    assert config["eval_sources"]
    assert all(source["eval_only"] for source in config["eval_sources"])


def test_weighted_bucket_mix_is_enforced_and_ids_are_stable(tmp_path: Path):
    text_manifest = tmp_path / "text.jsonl"
    image_manifest = tmp_path / "image.jsonl"
    text_manifest.write_text("".join(json.dumps({"id": f"t{i}", "prompt": f"text {i}"}) + "\n" for i in range(10)))
    image_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"i{i}",
                    "prompt": f"image {i}",
                    "image": f"image-{i}.png",
                }
            )
            + "\n"
            for i in range(10)
        )
    )
    config = {
        "recommended_mix": [
            {"bucket": "text", "share": 0.7, "required": True},
            {"bucket": "image", "share": 0.3, "required": True},
        ],
        "sources": [
            {
                "id": "text/source",
                "bucket": "text",
                "modality": "text",
                "manifest_only": True,
                "manifest_path": str(text_manifest),
            },
            {
                "id": "image/source",
                "bucket": "image",
                "modality": "image",
                "manifest_only": True,
                "manifest_path": str(image_manifest),
            },
        ],
    }

    rows = list(iter_prompt_records(config, None, 100, tmp_path, max_total_records=10))
    rows_again = list(iter_prompt_records(config, None, 100, tmp_path, max_total_records=10))

    assert [row["bucket"] for row in rows].count("text") == 7
    assert [row["bucket"] for row in rows].count("image") == 3
    assert [row["id"] for row in rows] == [row["id"] for row in rows_again]
    assert all(row["metadata"]["source_record_fingerprint"] for row in rows)


def test_required_bucket_underfill_is_fatal(tmp_path: Path):
    manifest = tmp_path / "only-one.jsonl"
    manifest.write_text(json.dumps({"id": "1", "prompt": "one"}) + "\n", encoding="utf-8")
    config = {
        "recommended_mix": [{"bucket": "text", "share": 1.0, "required": True}],
        "sources": [
            {
                "id": "tiny",
                "bucket": "text",
                "modality": "text",
                "manifest_only": True,
                "manifest_path": str(manifest),
            }
        ],
    }

    with pytest.raises(SourceMixError, match="underfilled"):
        list(iter_prompt_records(config, None, 100, tmp_path, max_total_records=2))


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "A valid teacher answer."}}]}


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse()


def test_openai_teacher_always_uses_chat_completions():
    cfg = TeacherConfig("openai-compatible", "teacher", "http://teacher/v1", 64, 0.2, 0.95)
    client = OpenAICompletionsClient(cfg)
    client.session = _FakeSession()

    assert client.generate("hello") == "A valid teacher answer."
    url, request = client.session.calls[0]
    assert url.endswith("/chat/completions")
    assert request["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert "prompt" not in request["json"]


def test_teacher_progress_fingerprint_mismatch_is_fatal(tmp_path: Path):
    output = tmp_path / "teacher.jsonl"
    output.write_text(
        json.dumps(
            {
                "id": "r1",
                "estimated_tokens": 4,
                "metadata": {"generation_fingerprint": "old", "source_index": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        read_progress(tmp_path / "progress.json", output, expected_fingerprint="new")


class _FakeTokenizer:
    pad_token_id = 0
    unk_token_id = 1
    eos_token_id = 99
    vocab_size = 128

    def __len__(self):
        return self.vocab_size

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        suffix = "<model>" if add_generation_prompt else ""
        return f"<chat>{messages[0]['content']}</chat>{suffix}"

    def encode(self, text, add_special_tokens=False):
        return [10 + (ord(char) % 50) for char in text]

    def decode(self, ids, **kwargs):
        return "<prior:" + ",".join(str(item) for item in ids) + ">"

    def convert_tokens_to_ids(self, token):
        return 98 if token == "<end_of_turn>" else self.unk_token_id


def test_corruption_chat_prefix_carries_prior_canvas_and_targets_end_with_eot():
    tokenizer = _FakeTokenizer()
    prefix = _chat_prefix_text(tokenizer, {"prompt_text": "question"}, [20, 21])
    blocks = list(_target_blocks(tokenizer, "ab", canvas_length=8))

    assert prefix.startswith("<chat>question</chat><model>")
    assert prefix.endswith("<prior:20,21>")
    assert blocks[0][:3].tolist()[-1] == 98
    assert 99 not in blocks[0]


def test_build_shards_writes_exact_manifest_and_provenance(tmp_path: Path, monkeypatch):
    from diffusiongemma_e4b import corruption

    raw = tmp_path / "teacher.jsonl"
    raw.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"record-{i}",
                    "prompt_source": "unit/source",
                    "prompt_text": f"question {i}",
                    "text": "abcdef",
                    "modality": "text",
                }
            )
            + "\n"
            for i in range(2)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        corruption.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(_commit_hash=None),
    )
    monkeypatch.setattr(corruption.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr(corruption.AutoProcessor, "from_pretrained", lambda *args, **kwargs: object())
    output = tmp_path / "corruption"

    manifest = build_shards(raw, output, "fake/student", target_blocks=3, canvas_length=4, prefix_length=16, shard_blocks=2)
    verified = verify_shards(output, min_blocks=3, canvas_length=4, expected_blocks=3)

    assert manifest["blocks_written"] == 3
    assert verified["exact"] is True
    with np.load(output / manifest["files"][0]["name"], allow_pickle=False) as shard:
        assert "record_ids" in shard
        assert "source_ids" in shard
        assert "chunk_index" in shard
        assert "modality" in shard
        assert shard["target_ids"].shape[1] == 4

    # An identical invocation verifies and reuses the completed fingerprint.
    reused = build_shards(raw, output, "fake/student", target_blocks=3, canvas_length=4, prefix_length=16, shard_blocks=2)
    assert reused["fingerprint"] == manifest["fingerprint"]
