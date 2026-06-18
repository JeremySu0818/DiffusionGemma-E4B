from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

from .constants import CANVAS_LENGTH


def entropy_from_logits(logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = logits.float() / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    log_probs = torch.log_softmax(scaled, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return probs, entropy


def entropy_bound_step(
    logits: torch.Tensor,
    current_canvas: torch.Tensor,
    entropy_bound: float,
    temperature: float,
    vocab_size: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs, entropy = entropy_from_logits(logits, temperature)
    candidate = torch.argmax(probs, dim=-1)
    accept = entropy <= entropy_bound
    if not bool(accept.any()):
        min_idx = torch.argmin(entropy, dim=-1)
        accept = torch.zeros_like(accept)
        accept.scatter_(1, min_idx[:, None], True)
    random_tokens = torch.randint(0, vocab_size, current_canvas.shape, device=current_canvas.device, generator=generator)
    next_canvas = torch.where(accept, candidate, random_tokens)
    return next_canvas, candidate, entropy


@torch.no_grad()
def strict_diffusion_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    canvas_length: int,
    denoise_steps: int,
    entropy_bound: float,
    confidence_threshold: float,
    stability_steps: int,
    temperature: float,
    seed: int,
) -> dict:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    committed = input_ids
    blocks = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    vocab_size = model.config.text_config.vocab_size

    remaining = max_new_tokens
    while remaining > 0:
        canvas = torch.randint(0, vocab_size, (1, canvas_length), device=device, generator=generator)
        self_conditioning_logits = None
        last_argmax = None
        stable = 0
        trace = []
        for step in range(denoise_steps):
            out = model(
                input_ids=committed,
                attention_mask=torch.ones_like(committed),
                decoder_input_ids=canvas,
                self_conditioning_logits=self_conditioning_logits,
            )
            logits = out.logits
            canvas, argmax_canvas, entropy = entropy_bound_step(
                logits, canvas, entropy_bound, temperature, vocab_size, generator
            )
            self_conditioning_logits = logits.detach()
            mean_entropy = float(entropy.mean().detach().cpu())
            confidence = float((entropy <= entropy_bound).float().mean().detach().cpu())
            if last_argmax is not None and torch.equal(argmax_canvas, last_argmax):
                stable += 1
            else:
                stable = 0
            last_argmax = argmax_canvas.detach().clone()
            trace.append({"step": step + 1, "mean_entropy": mean_entropy, "accepted_fraction": confidence, "stable": stable})
            if mean_entropy <= confidence_threshold or stable >= stability_steps:
                break
        commit = last_argmax[:, : min(canvas_length, remaining)]
        committed = torch.cat([committed, commit], dim=1)
        blocks.append({"tokens": int(commit.numel()), "steps": len(trace), "trace": trace})
        remaining -= int(commit.numel())
    new_ids = committed[:, input_ids.shape[1] :]
    return {
        "text": tokenizer.decode(new_ids[0], skip_special_tokens=True),
        "blocks": blocks,
        "strict_diffusion": True,
        "ar_fallback_used": False,
        "canvas_length": canvas_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/strict_diffusion_inference.json"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--denoise-steps", type=int, default=32)
    parser.add_argument("--entropy-bound", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--stability-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model_dir,
        dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    result = strict_diffusion_generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.canvas_length,
        args.denoise_steps,
        args.entropy_bound,
        args.confidence_threshold,
        args.stability_steps,
        args.temperature,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "blocks": len(result["blocks"])}, indent=2))


if __name__ == "__main__":
    main()
