from __future__ import annotations

import torch

from diffusiongemma_e4b.student import make_transplant_state_dict


def test_transplant_maps_text_vision_and_projection_families():
    base_state = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones((2, 2)),
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": torch.full((2, 2), 2.0),
        "model.embed_vision.embedding_projection.weight": torch.full((2, 2), 3.0),
        "model.audio_tower.encoder.layers.0.self_attn.q_proj.linear.weight": torch.full((2, 2), 4.0),
        "model.embed_audio.embedding_projection.weight": torch.full((2, 2), 5.0),
        "lm_head.weight": torch.full((2, 2), 6.0),
    }
    diffusion_state = {
        "model.encoder.language_model.layers.0.self_attn.q_proj.weight": torch.zeros((2, 2)),
        "model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": torch.zeros((2, 2)),
        "model.encoder.embed_vision.embedding_projection.weight": torch.zeros((2, 2)),
        "model.encoder.audio_tower.encoder.layers.0.self_attn.q_proj.linear.weight": torch.zeros((2, 2)),
        "model.encoder.embed_audio.embedding_projection.weight": torch.zeros((2, 2)),
        "lm_head.weight": torch.zeros((2, 2)),
    }

    mapped, report = make_transplant_state_dict(base_state, diffusion_state)

    assert torch.equal(
        mapped["model.encoder.language_model.layers.0.self_attn.q_proj.weight"],
        base_state["model.language_model.layers.0.self_attn.q_proj.weight"],
    )
    assert torch.equal(
        mapped["model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"],
        base_state["model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"],
    )
    assert torch.equal(
        mapped["model.encoder.embed_vision.embedding_projection.weight"],
        base_state["model.embed_vision.embedding_projection.weight"],
    )
    assert torch.equal(
        mapped["model.encoder.audio_tower.encoder.layers.0.self_attn.q_proj.linear.weight"],
        base_state["model.audio_tower.encoder.layers.0.self_attn.q_proj.linear.weight"],
    )
    assert torch.equal(
        mapped["model.encoder.embed_audio.embedding_projection.weight"],
        base_state["model.embed_audio.embedding_projection.weight"],
    )
    assert torch.equal(mapped["lm_head.weight"], base_state["lm_head.weight"])
    assert report["copied_count_by_family"]["text_encoder"] == 1
    assert report["copied_count_by_family"]["vision_tower"] == 1
    assert report["copied_count_by_family"]["vision_projector"] == 1
    assert report["copied_count_by_family"]["audio_tower"] == 1
    assert report["copied_count_by_family"]["audio_projector"] == 1
    assert report["copied_count_by_family"]["lm_head"] == 1
    assert report["image_text_target_present"] is True
    assert report["image_text_fully_copied"] is True
    assert report["audio_text_target_present"] is True
    assert report["audio_text_fully_copied"] is True


def test_transplant_reports_multimodal_shape_mismatch():
    base_state = {
        "model.vision_tower.patch_embedder.input_proj.weight": torch.ones((3, 3)),
        "model.audio_tower.subsample_conv_projection.layer0.conv.weight": torch.ones((3, 3)),
    }
    diffusion_state = {
        "model.encoder.vision_tower.patch_embedder.input_proj.weight": torch.zeros((2, 2)),
        "model.encoder.audio_tower.subsample_conv_projection.layer0.conv.weight": torch.zeros((2, 2)),
    }

    mapped, report = make_transplant_state_dict(base_state, diffusion_state)

    assert mapped == {}
    assert report["shape_mismatch_count_by_family"]["vision_tower"] == 1
    assert report["shape_mismatch_count_by_family"]["audio_tower"] == 1
    assert {item["family"] for item in report["shape_mismatch"]} == {"vision_tower", "audio_tower"}
