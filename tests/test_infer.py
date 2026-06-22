from __future__ import annotations

import torch

from diffusiongemma_e4b.infer import strict_diffusion_generate


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True, return_tensors="pt"):
        tokens = torch.tensor([[3, 4, 5]], dtype=torch.long)
        if return_tensors == "pt":
            return tokens
        return tokens[0].tolist()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in ids.tolist())


class FakeModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self.config = type("Config", (), {"text_config": type("TextConfig", (), {"vocab_size": vocab_size})()})()

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        batch, canvas = kwargs["decoder_input_ids"].shape
        logits = torch.zeros(batch, canvas, self.config.text_config.vocab_size, device=self.param.device)
        logits[..., 1] = 5.0
        return type("Output", (), {"logits": logits, "past_key_values": f"cache-{len(self.calls)}"})()


def test_strict_diffusion_generate_preserves_and_extends_attention_mask():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    encoder_inputs = {
        "input_ids": torch.tensor([[0, 0, 11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
    }

    strict_diffusion_generate(
        model,
        tokenizer,
        prompt="",
        max_new_tokens=4,
        canvas_length=2,
        denoise_steps=1,
        entropy_bound=0.0,
        confidence_threshold=-1.0,
        stability_steps=99,
        temperature=1.0,
        seed=7,
        encoder_inputs=encoder_inputs,
    )

    assert torch.equal(model.calls[0]["attention_mask"], torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    assert torch.equal(model.calls[1]["attention_mask"], torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long))


def test_strict_diffusion_generate_reuses_encoder_cache_within_block():
    model = FakeModel()
    tokenizer = FakeTokenizer()

    strict_diffusion_generate(
        model,
        tokenizer,
        prompt="hello",
        max_new_tokens=2,
        canvas_length=2,
        denoise_steps=2,
        entropy_bound=0.0,
        confidence_threshold=-1.0,
        stability_steps=99,
        temperature=1.0,
        seed=11,
    )

    assert "input_ids" in model.calls[0]
    assert "past_key_values" not in model.calls[0]
    assert "input_ids" not in model.calls[1]
    assert model.calls[1]["past_key_values"] == "cache-1"
