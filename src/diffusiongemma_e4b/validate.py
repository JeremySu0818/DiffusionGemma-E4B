from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

from .constants import CANVAS_LENGTH


def validate_artifact_files(model_dir: Path) -> dict:
    required_any = ["model.safetensors", "adapter_model.safetensors", "pytorch_model.bin"]
    existing = {p.name for p in model_dir.iterdir()} if model_dir.exists() else set()
    has_weights = any(name in existing for name in required_any) or any(p.name.startswith("model-") and p.suffix == ".safetensors" for p in model_dir.glob("*.safetensors"))
    return {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "has_config": (model_dir / "config.json").exists(),
        "has_tokenizer": any((model_dir / name).exists() for name in ["tokenizer.json", "tokenizer.model"]),
        "has_weights": has_weights,
        "files": sorted(existing),
    }


def validate_tokenizer(model_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return {
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
    }


def validate_forward(model_dir: Path, data_dir: Path, dtype: str, device_map: str) -> dict:
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_dir,
        dtype=getattr(torch, dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    shard_path = sorted(data_dir.glob("corruption_*.npz"))[0]
    with np.load(shard_path) as shard:
        prefix = torch.from_numpy(shard["prefix_ids"][:1].astype(np.int64))
        prefix_mask = torch.zeros_like(prefix)
        plen = int(shard["prefix_lens"][0])
        if plen:
            prefix_mask[:, -plen:] = 1
        corrupted = torch.from_numpy(shard["corrupted_ids"][:1].astype(np.int64))
        labels = torch.from_numpy(shard["target_ids"][:1].astype(np.int64))
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=prefix.to(device),
            attention_mask=prefix_mask.to(device),
            decoder_input_ids=corrupted.to(device),
        )
    logits = out.logits
    loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.to(device).view(-1))
    return {
        "forward_ok": True,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "canvas_length": int(logits.shape[1]),
    }


def fake_wrapper_exclusion(project_root: Path) -> dict:
    infer_path = project_root / "src" / "diffusiongemma_e4b" / "infer.py"
    train_path = project_root / "src" / "diffusiongemma_e4b" / "train.py"
    infer_src = infer_path.read_text(encoding="utf-8")
    train_src = train_path.read_text(encoding="utf-8")
    return {
        "strict_infer_uses_generate_api": ".generate(" in infer_src,
        "training_uses_diffusion_forward": "decoder_input_ids" in train_src and "cross_entropy" in train_src,
        "strict_infer_declares_no_ar_fallback": '"ar_fallback_used": False' in infer_src,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/corruption"))
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/validation_report.json"))
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    report = {
        "artifact_files": validate_artifact_files(args.model_dir),
        "fake_wrapper_exclusion": fake_wrapper_exclusion(Path.cwd()),
    }
    if report["artifact_files"]["has_tokenizer"]:
        report["tokenizer"] = validate_tokenizer(args.model_dir)
    if not args.skip_forward:
        report["forward"] = validate_forward(args.model_dir, args.data_dir, args.dtype, args.device_map)
    report["requirements"] = {
        "canvas_length_required": CANVAS_LENGTH,
        "strict_diffusion_not_ar_fallback": not report["fake_wrapper_exclusion"]["strict_infer_uses_generate_api"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
