from __future__ import annotations

import torch
from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig

from diffusiongemma_e4b.config import build_diffusion_e4b_config
from diffusiongemma_e4b.train import CorruptionShardDataset, compute_loss


def test_config_preserves_audio_fields(tmp_path):
    cfg = build_diffusion_e4b_config("google/gemma-4-E4B-it")
    cfg.save_pretrained(tmp_path)

    loaded = DiffusionGemmaConfig.from_pretrained(tmp_path)

    assert getattr(loaded, "audio_config", None) is not None
    assert getattr(loaded, "audio_token_id", None) == 258881
    assert loaded.architectures == ["MultimodalDiffusionGemmaForBlockDiffusion"]


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self.config = type("Config", (), {"text_config": type("Text", (), {"vocab_size": 8})()})()

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        batch, canvas = kwargs["decoder_input_ids"].shape
        logits = torch.zeros(batch, canvas, self.config.text_config.vocab_size, device=self.param.device)
        return type("Output", (), {"logits": logits})()


def test_compute_loss_passes_multimodal_kwargs_through_self_conditioning():
    model = RecordingModel()
    batch = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "decoder_input_ids": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.ones((1, 3), dtype=torch.long),
        "pixel_values": torch.ones((1, 3, 4, 4)),
        "input_features": torch.ones((1, 8, 4)),
        "input_features_mask": torch.ones((1, 8)),
        "image_position_ids": torch.zeros((1, 4, 2), dtype=torch.long),
        "mm_token_type_ids": torch.zeros((1, 4), dtype=torch.long),
    }

    loss = compute_loss(model, batch, self_conditioning_prob=1.0)

    assert loss.isfinite()
    assert len(model.calls) == 2
    for call in model.calls:
        assert "pixel_values" in call
        assert "input_features" in call
        assert "input_features_mask" in call
        assert "image_position_ids" in call
        assert "mm_token_type_ids" in call


def test_corruption_dataset_reads_multimodal_npz_fields(tmp_path):
    shard = tmp_path / "corruption_image_000000.npz"
    torch.manual_seed(0)
    import numpy as np

    np.savez_compressed(
        shard,
        prefix_ids=np.ones((2, 8), dtype=np.int64),
        prefix_lens=np.array([8, 8], dtype=np.int64),
        target_ids=np.ones((2, 4), dtype=np.int64),
        corrupted_ids=np.ones((2, 4), dtype=np.int64),
        corruption_masks=np.ones((2, 4), dtype=np.bool_),
        noise_t=np.array([0.1, 0.2], dtype=np.float32),
        pixel_values=np.ones((2, 3, 8, 8), dtype=np.float32),
        image_position_ids=np.zeros((2, 4, 2), dtype=np.int64),
    )

    ds = CorruptionShardDataset(tmp_path, split="train", val_fraction=0.5)
    item = ds[0]

    assert "pixel_values" in item
    assert "image_position_ids" in item
    assert item["pixel_values"].shape == (3, 8, 8)
