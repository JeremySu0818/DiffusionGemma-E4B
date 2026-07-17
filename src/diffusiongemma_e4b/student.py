from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil

import torch
import transformers
from accelerate import init_empty_weights
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    DiffusionGemmaGenerationConfig,
    EntropyBoundSamplerConfig,
)

from .config import (
    build_diffusion_e4b_config,
    build_diffusion_e4b_config_from_base_config,
)
from .constants import CANVAS_LENGTH, DEFAULT_BASE_MODEL
from .modeling_multimodal import MultimodalDiffusionGemmaForBlockDiffusion


BASE_PREFIXES = (
    "language_model.model.",
    "model.language_model.",
    "model.",
    "language_model.",
)


def _family_for_target(key: str) -> str:
    if key.startswith("model.encoder.language_model."):
        return "text_encoder"
    if key.startswith("model.encoder.vision_tower."):
        return "vision_tower"
    if key.startswith("model.encoder.embed_vision."):
        return "vision_projector"
    if key.startswith("model.encoder.audio_tower."):
        return "audio_tower"
    if key.startswith("model.encoder.embed_audio."):
        return "audio_projector"
    if key.startswith("model.decoder."):
        return "diffusion_decoder"
    if key == "lm_head.weight":
        return "lm_head"
    return "unsupported"


def create_diffusion_e4b_model(
    base_model: str = DEFAULT_BASE_MODEL,
    canvas_length: int = CANVAS_LENGTH,
    dtype: str = "bfloat16",
    base_config=None,
) -> MultimodalDiffusionGemmaForBlockDiffusion:
    cfg = (
        build_diffusion_e4b_config_from_base_config(
            base_config,
            canvas_length=canvas_length,
            dtype=dtype,
        )
        if base_config is not None
        else build_diffusion_e4b_config(base_model, canvas_length, dtype)
    )
    model = MultimodalDiffusionGemmaForBlockDiffusion(cfg)
    model.generation_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=canvas_length,
        max_denoising_steps=48,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
        confidence_threshold=0.005,
        stability_threshold=1,
        t_min=0.4,
        t_max=0.8,
        bos_token_id=cfg.text_config.bos_token_id,
        pad_token_id=cfg.text_config.pad_token_id,
        eos_token_id=cfg.text_config.eos_token_id,
    )
    model.tie_weights()
    return model


def _strip_base_prefix(key: str) -> str | None:
    for prefix in BASE_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    if key.startswith("lm_head."):
        return key
    return None


def _layer_source(prefix: str, suffix: str, suffix_to_tensor: dict[str, torch.Tensor]) -> torch.Tensor | None:
    return suffix_to_tensor.get(f"{prefix}.{suffix}")


def _target_source_suffix(dkey: str) -> str | None:
    if dkey.startswith("model.encoder.language_model."):
        return dkey[len("model.encoder.language_model.") :]
    if dkey.startswith("model.encoder.vision_tower."):
        return dkey[len("model.encoder.") :]
    if dkey.startswith("model.encoder.embed_vision."):
        return dkey[len("model.encoder.") :]
    if dkey.startswith("model.encoder.audio_tower."):
        return dkey[len("model.encoder.") :]
    if dkey.startswith("model.encoder.embed_audio."):
        return dkey[len("model.encoder.") :]
    if dkey.startswith("model.decoder."):
        suffix = dkey[len("model.decoder.") :]
        self_conditioning_sources = {
            "self_conditioning.pre_norm.weight": "layers.0.pre_feedforward_layernorm.weight",
            "self_conditioning.gate_proj.weight": "layers.0.mlp.gate_proj.weight",
            "self_conditioning.up_proj.weight": "layers.0.mlp.up_proj.weight",
            "self_conditioning.down_proj.weight": "layers.0.mlp.down_proj.weight",
        }
        return self_conditioning_sources.get(suffix, suffix)
    if dkey == "lm_head.weight":
        return "lm_head.weight"
    return None


def _empty_family_counts() -> dict[str, int]:
    return {
        "text_encoder": 0,
        "vision_tower": 0,
        "vision_projector": 0,
        "audio_tower": 0,
        "audio_projector": 0,
        "diffusion_decoder": 0,
        "lm_head": 0,
    }


def _add_family_count(counts: dict[str, int], family: str) -> None:
    if family in counts:
        counts[family] += 1


def make_transplant_state_dict(base_state: dict[str, torch.Tensor], diffusion_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    diffusion_keys = sorted(diffusion_state.keys())
    suffix_to_tensor: dict[str, torch.Tensor] = {}
    for key, tensor in base_state.items():
        suffix = _strip_base_prefix(key)
        if suffix is not None:
            suffix_to_tensor[suffix] = tensor

    mapped: dict[str, torch.Tensor] = {}
    report = {
        "copied": [],
        "missing_source": [],
        "shape_mismatch": [],
        "target_count_by_family": _empty_family_counts(),
        "copied_count_by_family": _empty_family_counts(),
        "missing_source_count_by_family": _empty_family_counts(),
        "shape_mismatch_count_by_family": _empty_family_counts(),
    }

    for dkey in diffusion_keys:
        family = _family_for_target(dkey)
        source_suffix = _target_source_suffix(dkey)
        if source_suffix is None:
            continue
        _add_family_count(report["target_count_by_family"], family)
        tensor = suffix_to_tensor.get(source_suffix)
        if tensor is None and source_suffix == "embed_tokens.weight":
            tensor = suffix_to_tensor.get("embed_tokens.weight")
        if tensor is None:
            report["missing_source"].append({"target": dkey, "source_suffix": source_suffix, "family": family})
            _add_family_count(report["missing_source_count_by_family"], family)
            continue
        if tuple(tensor.shape) != tuple(diffusion_state[dkey].shape):
            report["shape_mismatch"].append(
                {
                    "target": dkey,
                    "source_suffix": source_suffix,
                    "source_shape": list(tensor.shape),
                    "target_shape": list(diffusion_state[dkey].shape),
                    "family": family,
                }
            )
            _add_family_count(report["shape_mismatch_count_by_family"], family)
            continue
        # These targets intentionally start from E4B values but are not tied
        # weights. With assign=True, reusing the same storage would make
        # safetensors reject the model and would also couple self-conditioning
        # updates to the base MLP.
        if dkey.startswith("model.decoder.self_conditioning.") or (
            dkey.startswith("model.decoder.layers.")
            and dkey.endswith(".layer_scalar")
        ):
            tensor = tensor.clone()
        mapped[dkey] = tensor
        report["copied"].append({"target": dkey, "source_suffix": source_suffix, "shape": list(tensor.shape), "family": family})
        _add_family_count(report["copied_count_by_family"], family)
    report["coverage_by_family"] = {
        family: round(report["copied_count_by_family"][family] / total, 6) if total else None
        for family, total in report["target_count_by_family"].items()
    }
    report["image_text_target_present"] = (
        report["target_count_by_family"]["vision_tower"] > 0
        and report["target_count_by_family"]["vision_projector"] > 0
    )
    report["image_text_fully_copied"] = (
        report["image_text_target_present"]
        and report["coverage_by_family"]["vision_tower"] == 1.0
        and report["coverage_by_family"]["vision_projector"] == 1.0
    )
    report["audio_text_target_present"] = (
        report["target_count_by_family"]["audio_tower"] > 0
        and report["target_count_by_family"]["audio_projector"] > 0
    )
    report["audio_text_fully_copied"] = (
        report["audio_text_target_present"]
        and report["coverage_by_family"]["audio_tower"] == 1.0
        and report["coverage_by_family"]["audio_projector"] == 1.0
    )
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


def validate_transplant_report(report: dict) -> None:
    problems = []
    if report.get("missing_source_count", 0) > 0:
        problems.append(f"missing_source_count={report['missing_source_count']}")
    if report.get("shape_mismatch"):
        problems.append(f"shape_mismatch_count={len(report['shape_mismatch'])}")
    if report.get("load_missing"):
        problems.append(f"load_missing_count={len(report['load_missing'])}")
    if report.get("load_unexpected"):
        problems.append(f"load_unexpected_count={len(report['load_unexpected'])}")
    if report.get("image_text_target_present") and not report.get("image_text_fully_copied"):
        problems.append("image_text_partial_copy")
    if report.get("audio_text_target_present") and not report.get("audio_text_fully_copied"):
        problems.append("audio_text_partial_copy")
    if problems:
        raise RuntimeError(
            "Weight transplant coverage is incomplete; refusing to continue. "
            + "Problems: "
            + ", ".join(problems)
            + ". Inspect weight_transplant_report.json; production does not permit partial E4B weights."
        )


def _indexed_weight_files(output_dir: Path) -> list[Path]:
    index_path = output_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set((payload.get("weight_map") or {}).values()))
        return [output_dir / name for name in names]
    candidates = sorted(output_dir.glob("model*.safetensors"))
    if not candidates:
        candidates = sorted(output_dir.glob("pytorch_model*.bin"))
    return candidates


def _reusable_transplant(
    output_dir: Path,
    base_model: str,
    canvas_length: int,
    dtype: str,
) -> dict | None:
    report_path = output_dir / "weight_transplant_report.json"
    config_path = output_dir / "config.json"
    if not report_path.is_file() or not config_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    weights = _indexed_weight_files(output_dir)
    if not weights or any(not path.is_file() or path.stat().st_size <= 0 for path in weights):
        return None
    text_config = config.get("text_config") or {}
    if (
        report.get("base_model") != base_model
        or int(report.get("canvas_length", -1)) != canvas_length
        or report.get("dtype") != dtype
        or report.get("missing_source_count") != 0
        or report.get("shape_mismatch")
        or report.get("load_missing")
        or report.get("load_unexpected")
        or config.get("architectures")
        != ["MultimodalDiffusionGemmaForBlockDiffusion"]
        or int(text_config.get("hidden_size_per_layer_input") or 0) <= 0
        or bool(text_config.get("enable_moe_block"))
    ):
        return None
    report["reused"] = True
    report["weight_files"] = [path.name for path in weights]
    return report


def transplant_weights(
    base_model: str,
    output_dir: Path,
    canvas_length: int = CANVAS_LENGTH,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
    save_full_model: bool = True,
    force_rebuild: bool = False,
) -> dict:
    if save_full_model and not force_rebuild:
        reusable = _reusable_transplant(
            output_dir, base_model, canvas_length, dtype
        )
        if reusable is not None:
            return reusable

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    base = load_base_model(base_model, dtype=dtype, device_map=device_map)
    base_revision = getattr(base.config, "_commit_hash", None)
    # The encoder and decoder intentionally share every compatible E4B weight.
    # Constructing both branches normally would nevertheless allocate both
    # copies before tie_weights() runs, which can exceed host RAM. Build only
    # shape metadata here and materialize directly from the exact source config
    # that accompanied the loaded weights.
    with init_empty_weights():
        student = create_diffusion_e4b_model(
            base_model,
            canvas_length,
            dtype,
            base_config=base.config,
        )
    mapped, report = make_transplant_state_dict(base.state_dict(), student.state_dict())
    missing, unexpected = student.load_state_dict(
        mapped,
        strict=False,
        assign=True,
    )
    student.tie_weights()
    report["load_missing"] = list(missing)
    report["load_unexpected"] = list(unexpected)
    report["copied_count"] = len(report["copied"])
    report["missing_source_count"] = len(report["missing_source"])
    report["base_model"] = base_model
    report["base_model_revision"] = base_revision
    report["canvas_length"] = canvas_length
    report["dtype"] = dtype
    report["student_architecture"] = "MultimodalDiffusionGemmaForBlockDiffusion"
    report["reused"] = False
    del mapped, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    report_path = staging / "weight_transplant_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    validate_transplant_report(report)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        revision=base_revision,
    )
    tokenizer.save_pretrained(staging)
    try:
        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            revision=base_revision,
        )
        processor.save_pretrained(staging)
    except Exception as exc:  # noqa: BLE001
        (staging / "processor_save_warning.txt").write_text(
            str(exc), encoding="utf-8"
        )

    student.config.save_pretrained(staging)
    (staging / "diffusiongemma_e4b_config_metadata.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "base_model_revision": report["base_model_revision"],
                "canvas_length": canvas_length,
                "dtype": dtype,
                "student_architecture": (
                    "diffusiongemma_e4b.modeling_multimodal."
                    "MultimodalDiffusionGemmaForBlockDiffusion"
                ),
                "conversion": (
                    "Gemma 4 E4B weights transplanted into an E4B-sized "
                    "block-diffusion model"
                ),
                "production_supported": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if save_full_model:
        student.save_pretrained(
            staging,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        weights = _indexed_weight_files(staging)
        if not weights or any(path.stat().st_size <= 0 for path in weights):
            raise RuntimeError("transplant save did not produce complete model weights")

    backup = output_dir.parent / f".{output_dir.name}.previous-{os.getpid()}"
    if backup.exists():
        shutil.rmtree(backup)
    if output_dir.exists():
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        if backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transplanted"))
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-save-full-model", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    report = transplant_weights(
        args.base_model,
        args.output_dir,
        args.canvas_length,
        args.dtype,
        None if args.device_map == "none" else args.device_map,
        save_full_model=not args.no_save_full_model,
        force_rebuild=args.force_rebuild,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "copied_count",
                    "missing_source_count",
                    "base_model",
                    "coverage_by_family",
                    "image_text_target_present",
                    "image_text_fully_copied",
                    "audio_text_target_present",
                    "audio_text_fully_copied",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
