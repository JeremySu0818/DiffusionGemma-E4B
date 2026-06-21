from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffusiongemma_e4b.data_contract import TeacherSupervisedRecord, write_teacher_jsonl
from diffusiongemma_e4b.data_sources import _prompt_record
from diffusiongemma_e4b.teacher import _chat_content


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
