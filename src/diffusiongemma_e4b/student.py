from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

from .config import build_diffusion_e4b_config
from .constants import CANVAS_LENGTH, DEFAULT_BASE_MODEL


BASE_PREFIXES = (
    "language_model.model.",
    "model.language_model.",
    "model.",
    "language_model.",
)


def create_diffusion_e4b_model(
    base_model: str = DEFAULT_BASE_MODEL,
    canvas_length: int = CANVAS_LENGTH,
    dtype: str = "bfloat16",
) -> DiffusionGemmaForBlockDiffusion:
    cfg = build_diffusion_e4b_config(base_model, canvas_length, dtype)
    return DiffusionGemmaForBlockDiffusion(cfg)


def _strip_base_prefix(key: str) -> str | None:
    for prefix in BASE_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    if key.startswith("lm_head."):
        return key
    return None


def _layer_source(prefix: str, suffix: str, suffix_to_tensor: dict[str, torch.Tensor]) -> torch.Tensor | None:
    return suffix_to_tensor.get(f"{prefix}.{suffix}")


def make_transplant_state_dict(base_state: dict[str, torch.Tensor], diffusion_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    diffusion_keys = set(diffusion_state.keys())
    suffix_to_tensor: dict[str, torch.Tensor] = {}
    for key, tensor in base_state.items():
        suffix = _strip_base_prefix(key)
        if suffix is not None:
            suffix_to_tensor[suffix] = tensor

    mapped: dict[str, torch.Tensor] = {}
    report = {"copied": [], "missing_source": [], "shape_mismatch": []}

    for dkey in diffusion_keys:
        source_suffix = None
        if dkey.startswith("model.encoder.language_model."):
            source_suffix = dkey[len("model.encoder.language_model.") :]
        elif dkey.startswith("model.decoder."):
            source_suffix = dkey[len("model.decoder.") :]
        elif dkey == "lm_head.weight":
            source_suffix = "lm_head.weight"

        if source_suffix is None:
            continue
        tensor = suffix_to_tensor.get(source_suffix)
        if tensor is None and ".experts.gate_up_proj" in source_suffix:
            layer_prefix = source_suffix.split(".experts.gate_up_proj")[0]
            gate = _layer_source(f"{layer_prefix}.mlp", "gate_proj.weight", suffix_to_tensor)
            up = _layer_source(f"{layer_prefix}.mlp", "up_proj.weight", suffix_to_tensor)
            if gate is not None and up is not None:
                tensor = torch.cat([gate, up], dim=0).unsqueeze(0).to(diffusion_state[dkey].dtype)
        if tensor is None and ".experts.down_proj" in source_suffix:
            layer_prefix = source_suffix.split(".experts.down_proj")[0]
            down = _layer_source(f"{layer_prefix}.mlp", "down_proj.weight", suffix_to_tensor)
            if down is not None:
                tensor = down.unsqueeze(0).to(diffusion_state[dkey].dtype)
        if tensor is None and source_suffix.endswith(".router.proj.weight"):
            tensor = torch.zeros_like(diffusion_state[dkey])
        if tensor is None and source_suffix.endswith(".router.per_expert_scale"):
            tensor = torch.ones_like(diffusion_state[dkey])
        if tensor is None and source_suffix == "embed_tokens.weight":
            tensor = suffix_to_tensor.get("embed_tokens.weight")
        if tensor is None:
            report["missing_source"].append({"target": dkey, "source_suffix": source_suffix})
            continue
        if tuple(tensor.shape) != tuple(diffusion_state[dkey].shape):
            report["shape_mismatch"].append(
                {"target": dkey, "source_suffix": source_suffix, "source_shape": list(tensor.shape), "target_shape": list(diffusion_state[dkey].shape)}
            )
            continue
        mapped[dkey] = tensor
        report["copied"].append({"target": dkey, "source_suffix": source_suffix, "shape": list(tensor.shape)})
    return mapped, report


def load_base_model(model_id_or_path: str, dtype: str, device_map: str | None):
    torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else "auto"
    errors = []
    class_candidates = [AutoModelForCausalLM]
    for name in ("AutoModelForImageTextToText", "AutoModelForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            class_candidates.append(cls)
    for cls in class_candidates:
        try:
            return cls.from_pretrained(
                model_id_or_path,
                dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError("Could not load base Gemma 4 E4B model:\n" + "\n".join(errors))


def transplant_weights(
    base_model: str,
    output_dir: Path,
    canvas_length: int = CANVAS_LENGTH,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
    save_full_model: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    student = create_diffusion_e4b_model(base_model, canvas_length, dtype)
    base = load_base_model(base_model, dtype=dtype, device_map=device_map)
    mapped, report = make_transplant_state_dict(base.state_dict(), student.state_dict())
    missing, unexpected = student.load_state_dict(mapped, strict=False)
    report["load_missing"] = list(missing)
    report["load_unexpected"] = list(unexpected)
    report["copied_count"] = len(report["copied"])
    report["missing_source_count"] = len(report["missing_source"])
    report["base_model"] = base_model
    report["canvas_length"] = canvas_length
    report["dtype"] = dtype
    report_path = output_dir / "weight_transplant_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        processor.save_pretrained(output_dir)
    except Exception as exc:  # noqa: BLE001
        (output_dir / "processor_save_warning.txt").write_text(str(exc), encoding="utf-8")

    student.config.save_pretrained(output_dir)
    if save_full_model:
        student.save_pretrained(output_dir, safe_serialization=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transplanted"))
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-save-full-model", action="store_true")
    args = parser.parse_args()
    report = transplant_weights(
        args.base_model,
        args.output_dir,
        args.canvas_length,
        args.dtype,
        None if args.device_map == "none" else args.device_map,
        save_full_model=not args.no_save_full_model,
    )
    print(json.dumps({k: report[k] for k in ["copied_count", "missing_source_count", "base_model"]}, indent=2))


if __name__ == "__main__":
    main()
