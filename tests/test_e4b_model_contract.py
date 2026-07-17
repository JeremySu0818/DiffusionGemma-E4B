from __future__ import annotations

from types import SimpleNamespace
from dataclasses import asdict
import json

import pytest
import torch
from transformers import (
    DiffusionGemmaConfig,
    DiffusionGemmaGenerationConfig,
    DiffusionGemmaTextConfig,
    EntropyBoundSamplerConfig,
    Gemma4VisionConfig,
)

from diffusiongemma_e4b.modeling_multimodal import (
    MultimodalDiffusionGemmaForBlockDiffusion,
)
import diffusiongemma_e4b.student as student_module
from diffusiongemma_e4b.train import (
    DeterministicValidationBatchSampler,
    TrainState,
    _decoder_linear_targets,
    _reconcile_best_snapshot,
    _relative_validation_improvement,
    compute_loss,
    maybe_apply_lora,
)
from diffusiongemma_e4b.validate import validate_e4b_architecture


def _tiny_e4b_model() -> MultimodalDiffusionGemmaForBlockDiffusion:
    text = DiffusionGemmaTextConfig(
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size=32,
        hidden_size_per_layer_input=8,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=2,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=["full_attention", "sliding_attention"],
        num_kv_shared_layers=1,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        attention_k_eq_v=False,
    )
    vision = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        position_embedding_size=64,
    )
    config = DiffusionGemmaConfig(
        text_config=text,
        vision_config=vision,
        canvas_length=4,
    )
    config.architectures = ["MultimodalDiffusionGemmaForBlockDiffusion"]
    return MultimodalDiffusionGemmaForBlockDiffusion(config)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[2, 7, 8, 9]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "decoder_input_ids": torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
        "decoder_attention_mask": torch.ones((1, 4), dtype=torch.long),
        "labels": torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
        "corruption_mask": torch.tensor([[True, False, True, False]]),
    }


def test_e4b_forward_accepts_prefix_plus_canvas_mask() -> None:
    model = _tiny_e4b_model()
    loss = compute_loss(
        model,
        _batch(),
        self_conditioning_prob=1.0,
        clean_token_loss_weight=0.0,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_e4b_save_reload_keeps_tied_weights_and_finite_forward(tmp_path) -> None:
    model = _tiny_e4b_model().eval()
    model.save_pretrained(tmp_path, safe_serialization=True)

    loaded = MultimodalDiffusionGemmaForBlockDiffusion.from_pretrained(
        tmp_path
    ).eval()
    encoder = loaded.model.encoder.language_model
    decoder = loaded.model.decoder
    encoder_parameters = dict(encoder.named_parameters())
    decoder_parameters = dict(decoder.named_parameters())
    shared_names = sorted(set(encoder_parameters) & set(decoder_parameters))
    assert shared_names
    assert all(
        encoder_parameters[name] is decoder_parameters[name]
        for name in shared_names
    )
    assert loaded.lm_head.weight is decoder.embed_tokens.weight
    architecture = validate_e4b_architecture(loaded)
    assert architecture["comparable_encoder_decoder_parameters"] > 0
    assert architecture["untied_encoder_decoder_parameters"] == []
    assert architecture["lm_head_decoder_embedding_tied"] is True

    inputs = {
        key: value
        for key, value in _batch().items()
        if key not in {"labels", "corruption_mask"}
    }
    with torch.no_grad():
        output = loaded(**inputs)
    assert output.logits.shape == (1, 4, 128)
    assert torch.isfinite(output.logits).all()


def test_e4b_uses_strict_diffusion_generation() -> None:
    model = _tiny_e4b_model().eval()
    generation_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=4,
        max_denoising_steps=2,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
        confidence_threshold=0.005,
        stability_threshold=1,
        t_min=0.4,
        t_max=0.8,
        bos_token_id=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    with torch.no_grad():
        output = model.generate(
            input_ids=torch.tensor([[2, 7, 8]], dtype=torch.long),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            generation_config=generation_config,
        )
    assert type(output).__name__ == "DiffusionGemmaGenerationOutput"
    assert output.sequences.shape == (1, 7)


def test_transplant_materializes_from_meta_and_saves_reloadable_model(
    tmp_path,
    monkeypatch,
) -> None:
    reference = _tiny_e4b_model()
    reference.config._commit_hash = "tiny-revision"
    source_state = {}
    for target_name, tensor in reference.state_dict().items():
        source_suffix = student_module._target_source_suffix(target_name)
        assert source_suffix is not None
        source_state.setdefault(f"model.{source_suffix}", tensor.detach().clone())

    base = SimpleNamespace(
        config=reference.config,
        state_dict=lambda: source_state,
    )

    class SavableProcessor:
        def save_pretrained(self, _output_dir):
            return None

    monkeypatch.setattr(
        student_module,
        "load_base_model",
        lambda *_args, **_kwargs: base,
    )
    monkeypatch.setattr(
        student_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SavableProcessor(),
    )
    monkeypatch.setattr(
        student_module.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: SavableProcessor(),
    )

    output_dir = tmp_path / "transplanted"
    report = student_module.transplant_weights(
        "fake/e4b",
        output_dir,
        canvas_length=4,
        dtype="float32",
        device_map=None,
    )
    loaded = MultimodalDiffusionGemmaForBlockDiffusion.from_pretrained(
        output_dir
    )

    assert report["missing_source_count"] == 0
    assert report["shape_mismatch"] == []
    assert not any(parameter.device.type == "meta" for parameter in loaded.parameters())
    assert (
        loaded.model.encoder.language_model.embed_tokens.weight
        is loaded.model.decoder.embed_tokens.weight
    )
    self_conditioning_gate = loaded.model.decoder.self_conditioning.gate_proj.weight
    decoder_gate = loaded.model.decoder.layers[0].mlp.gate_proj.weight
    assert self_conditioning_gate is not decoder_gate
    assert torch.equal(self_conditioning_gate, decoder_gate)


def test_lora_targets_only_e4b_decoder_linears() -> None:
    model = _tiny_e4b_model()
    targets = _decoder_linear_targets(model, "auto")
    assert targets
    assert all(".decoder." in f".{name}." for name in targets)
    assert not any("encoder" in name or "vision_tower" in name or "lm_head" in name for name in targets)

    args = SimpleNamespace(
        train_mode="lora",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules="auto",
        model_dir="artifacts/transplanted",
    )
    adapted = maybe_apply_lora(model, args)
    trainable = [name for name, parameter in adapted.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_validation_sampler_is_uniform_complete_and_deterministic() -> None:
    class DatasetWithBuckets(list):
        bucket_labels = ["general"] * 20 + ["reasoning"] * 10 + ["image"]

    dataset = DatasetWithBuckets(range(31))
    first = list(
        DeterministicValidationBatchSampler(dataset, seed=17, max_samples=12)
    )
    second = list(
        DeterministicValidationBatchSampler(dataset, seed=17, max_samples=12)
    )
    assert first == second
    assert len(first) == 12
    assert len({index for batch in first for index in batch}) == 12
    assert all(len(batch) == 1 for batch in first)
    assert {
        dataset.bucket_labels[batch[0]] for batch in first
    } == {"general", "reasoning", "image"}


def test_validation_sampler_reallocates_quota_from_exhausted_bucket() -> None:
    class DatasetWithRareBucket(list):
        bucket_labels = ["general"] * 99 + ["image"]

    dataset = DatasetWithRareBucket(range(100))
    sampler = DeterministicValidationBatchSampler(
        dataset, seed=17, max_samples=100
    )
    selected = list(sampler)

    assert len(sampler) == 100
    assert len(selected) == 100
    assert len({index for batch in selected for index in batch}) == 100
    assert {
        dataset.bucket_labels[batch[0]] for batch in selected
    } == {"general", "image"}


def test_relative_validation_improvement_uses_pretraining_baseline() -> None:
    state = TrainState(baseline_val_loss=4.0, best_val_loss=3.6)
    assert _relative_validation_improvement(state) == pytest.approx(0.1)


def test_best_snapshot_newer_than_checkpoint_is_reconciled(tmp_path) -> None:
    current = TrainState(
        data_fingerprint="data",
        training_fingerprint="train",
        baseline_val_loss=4.0,
        best_val_loss=3.8,
        best_optimizer_steps=1_000,
    )
    newer = TrainState(
        data_fingerprint="data",
        training_fingerprint="train",
        baseline_val_loss=4.0,
        best_val_loss=3.6,
        best_optimizer_steps=1_500,
    )
    best = tmp_path / "best"
    best.mkdir()
    (best / "best_state.json").write_text(
        json.dumps(asdict(newer)),
        encoding="utf-8",
    )
    _reconcile_best_snapshot(current, tmp_path)
    assert current.best_val_loss == 3.6
    assert current.best_optimizer_steps == 1_500
