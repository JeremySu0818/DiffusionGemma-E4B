from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from .constants import CANVAS_LENGTH
from .infer import MULTIMODAL_BATCH_KEYS, _model_input_device, load_model_and_processor
from .train import corruption_data_fingerprint


def _package_versions() -> dict[str, str | None]:
    names = (
        "accelerate",
        "bitsandbytes",
        "datasets",
        "huggingface-hub",
        "numpy",
        "peft",
        "torch",
        "torchvision",
        "transformers",
        "vllm",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_artifact_files(model_source: str | Path) -> dict[str, Any]:
    model_dir = Path(str(model_source))
    if not model_dir.exists():
        return {
            "model_dir": str(model_source),
            "exists": False,
            "remote_model_id": True,
            "has_weights": False,
            "has_adapter": False,
            "has_processor": False,
        }

    files = sorted(path.name for path in model_dir.iterdir())
    has_adapter = (model_dir / "adapter_config.json").exists() and any(
        (model_dir / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    has_full_weights = any(
        (model_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin")
    ) or any(model_dir.glob("model-*.safetensors"))
    missing_weight_shards: list[str] = []
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_dir / index_name
        if not index_path.exists():
            continue
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = set((index_payload.get("weight_map") or {}).values())
        if not shard_names:
            raise ValueError(f"weight index has no weight_map entries: {index_path}")
        missing_weight_shards.extend(
            sorted(name for name in shard_names if not (model_dir / name).is_file())
        )
    if missing_weight_shards:
        raise FileNotFoundError(
            f"artifact is missing indexed weight shards: {sorted(set(missing_weight_shards))}"
        )
    has_processor = any(
        (model_dir / name).exists()
        for name in (
            "processor_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
        )
    )
    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "exists": True,
        "remote_model_id": False,
        "has_weights": bool(has_full_weights or has_adapter),
        "has_full_weights": bool(has_full_weights),
        "has_adapter": bool(has_adapter),
        "has_config": (model_dir / "config.json").exists(),
        "has_processor": has_processor,
        "has_generation_config": (model_dir / "generation_config.json").exists(),
        "has_artifact_metadata": any(
            (model_dir / name).exists()
            for name in ("artifact_metadata.json", "training_metadata.json")
        ),
        "indexed_weight_shards_complete": not missing_weight_shards,
        "files": files,
    }
    if (model_dir / "adapter_config.json").exists():
        adapter = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        report["adapter_base_model"] = adapter.get("base_model_name_or_path")
    return report


def validate_processor(processor) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_ids = list(eos_token_id) if isinstance(eos_token_id, (tuple, list)) else eos_token_id
    return {
        "class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": eos_ids,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "has_chat_template": bool(getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)),
    }


def validate_shards(data_dir: Path, expected_canvas_length: int = CANVAS_LENGTH) -> dict[str, Any]:
    files = sorted(data_dir.glob("corruption_*.npz"))
    if not files:
        raise FileNotFoundError(f"no corruption_*.npz files in {data_dir}")

    total_rows = 0
    corrupted_tokens = 0
    valid_tokens = 0
    record_ids: set[str] = set()
    families: dict[str, int] = {}
    buckets: dict[str, int] = {}
    noise_min = math.inf
    noise_max = -math.inf
    for path in files:
        with np.load(path, allow_pickle=False) as shard:
            required = {"prefix_ids", "prefix_lens", "target_ids", "corrupted_ids", "corruption_masks", "noise_t"}
            missing = sorted(required.difference(shard.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {missing}")
            target = shard["target_ids"]
            corrupted = shard["corrupted_ids"]
            corruption_mask = shard["corruption_masks"].astype(bool)
            if target.ndim != 2 or target.shape[1] != expected_canvas_length:
                raise ValueError(f"{path} target shape {target.shape} does not match canvas {expected_canvas_length}")
            if corrupted.shape != target.shape or corruption_mask.shape != target.shape:
                raise ValueError(f"{path} target/corrupted/mask shapes differ")
            rows = int(target.shape[0])
            if shard["prefix_ids"].shape[0] != rows or shard["prefix_lens"].shape[0] != rows:
                raise ValueError(f"{path} prefix row count differs from targets")
            valid = target != -100
            if bool((corruption_mask & ~valid).any()):
                raise ValueError(f"{path} corrupts padding/ignored target positions")
            if bool((corruption_mask.sum(axis=1) == 0).any()):
                raise ValueError(f"{path} contains a row with zero denoising targets")
            if bool((valid.sum(axis=1) == 0).any()):
                raise ValueError(f"{path} contains an empty target row")
            for key in ("attention_mask", "attention_masks", *MULTIMODAL_BATCH_KEYS):
                if key in shard and shard[key].shape[0] != rows:
                    raise ValueError(f"{path} {key} row count differs from targets")
            if "record_ids" in shard:
                record_ids.update(str(value) for value in shard["record_ids"].tolist())
            if "buckets" in shard:
                for value in shard["buckets"].tolist():
                    bucket = str(value)
                    buckets[bucket] = buckets.get(bucket, 0) + 1
            noise = shard["noise_t"].astype(np.float32)
            if bool(((noise <= 0.0) | (noise >= 1.0)).any()):
                raise ValueError(f"{path} contains noise_t outside (0, 1)")
            noise_min = min(noise_min, float(noise.min()))
            noise_max = max(noise_max, float(noise.max()))
            total_rows += rows
            valid_tokens += int(valid.sum())
            corrupted_tokens += int(corruption_mask.sum())
            family = path.stem.split("_")[1] if "_" in path.stem else "unknown"
            families[family] = families.get(family, 0) + rows

    return {
        "files": len(files),
        "blocks": total_rows,
        "valid_tokens": valid_tokens,
        "corrupted_tokens": corrupted_tokens,
        "corrupted_fraction": corrupted_tokens / max(valid_tokens, 1),
        "unique_record_ids": len(record_ids) if record_ids else None,
        "record_level_split_supported": bool(record_ids),
        "noise_t_min": noise_min,
        "noise_t_max": noise_max,
        "families": families,
        "buckets": buckets,
        "bucket_block_shares": {
            bucket: count / total_rows
            for bucket, count in sorted(buckets.items())
        },
        "canvas_length": expected_canvas_length,
    }


def _first_batch(data_dir: Path) -> dict[str, torch.Tensor]:
    shard_path = sorted(data_dir.glob("corruption_*.npz"))[0]
    with np.load(shard_path, allow_pickle=False) as shard:
        prefix = torch.from_numpy(shard["prefix_ids"][:1].astype(np.int64))
        attention_mask_key = next(
            (key for key in ("attention_mask", "attention_masks") if key in shard),
            None,
        )
        if attention_mask_key is not None:
            prefix_mask = torch.from_numpy(shard[attention_mask_key][:1].astype(np.int64))
        else:
            prefix_mask = torch.zeros_like(prefix)
            prefix_len = int(shard["prefix_lens"][0])
            if prefix_len:
                prefix_mask[:, -prefix_len:] = 1
        target = torch.from_numpy(shard["target_ids"][:1].astype(np.int64))
        batch: dict[str, torch.Tensor] = {
            "input_ids": prefix,
            "attention_mask": prefix_mask,
            "decoder_input_ids": torch.from_numpy(shard["corrupted_ids"][:1].astype(np.int64)),
            "decoder_attention_mask": torch.cat([prefix_mask, target.ne(-100).long()], dim=1),
            "labels": target,
            "corruption_mask": torch.from_numpy(shard["corruption_masks"][:1].astype(bool)),
        }
        for key in MULTIMODAL_BATCH_KEYS:
            if key not in shard:
                continue
            value = shard[key][:1]
            if key in {"pixel_values", "input_features"}:
                batch[key] = torch.from_numpy(value.astype(np.float32))
            else:
                batch[key] = torch.from_numpy(value.astype(np.int64))
    return batch


def validate_forward(model, data_dir: Path) -> dict[str, Any]:
    batch = _first_batch(data_dir)
    device = _model_input_device(model)
    model_inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key not in {"labels", "corruption_mask"}
    }
    model.eval()
    with torch.no_grad():
        output = model(**model_inputs)
    logits = output.logits.float()
    labels = batch["labels"].to(logits.device)
    corruption_mask = batch["corruption_mask"].to(logits.device) & labels.ne(-100)
    safe_labels = labels.masked_fill(labels.eq(-100), 0)
    token_loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2),
        safe_labels,
        reduction="none",
    )
    masked_loss = (token_loss * corruption_mask).sum() / corruption_mask.sum().clamp_min(1)
    return {
        "forward_ok": True,
        "model_class": type(model).__name__,
        "logits_shape": list(logits.shape),
        "masked_denoising_loss": float(masked_loss.detach().cpu()),
        "finite_loss": bool(torch.isfinite(masked_loss)),
        "canvas_length": int(logits.shape[1]),
    }


def validate_generation_contract(model) -> dict[str, Any]:
    config = getattr(model, "generation_config", None)
    model_config = getattr(model, "config", None)
    sampler = getattr(config, "sampler_config", None) if config is not None else None
    if isinstance(sampler, dict):
        entropy_bound = sampler.get("entropy_bound")
        sampler_class = sampler.get("_cls_name")
    else:
        entropy_bound = getattr(sampler, "entropy_bound", None)
        sampler_class = type(sampler).__name__ if sampler is not None else None
    has_generate = callable(getattr(model, "generate", None))
    model_type = getattr(model_config, "model_type", None)
    generation_config_class = type(config).__name__ if config is not None else None
    max_denoising_steps = getattr(config, "max_denoising_steps", None)
    confidence_threshold = getattr(config, "confidence_threshold", None)
    stability_threshold = getattr(config, "stability_threshold", None)
    t_min = getattr(config, "t_min", None)
    t_max = getattr(config, "t_max", None)
    strict = all(
        (
            has_generate,
            model_type == "diffusion_gemma",
            generation_config_class == "DiffusionGemmaGenerationConfig",
            sampler_class == "EntropyBoundSamplerConfig",
            isinstance(max_denoising_steps, int) and max_denoising_steps > 0,
            entropy_bound is not None,
            confidence_threshold is not None,
            stability_threshold is not None,
            t_min is not None,
            t_max is not None,
        )
    )
    return {
        "has_generate": has_generate,
        "model_type": model_type,
        "canvas_length": getattr(model_config, "canvas_length", None),
        "generation_config_class": generation_config_class,
        "sampler_config_class": sampler_class,
        "max_denoising_steps": max_denoising_steps,
        "confidence_threshold": confidence_threshold,
        "stability_threshold": stability_threshold,
        "entropy_bound": entropy_bound,
        "t_min": t_min,
        "t_max": t_max,
        "strict_diffusion_not_ar_fallback": strict,
    }


def validate_e4b_architecture(model) -> dict[str, Any]:
    base = model
    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()
        except Exception:  # noqa: BLE001
            base = model
    config = getattr(base, "config", getattr(model, "config", None))
    text_config = getattr(config, "text_config", None)
    expected = {
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_hidden_layers": 42,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "num_kv_shared_layers": 18,
        "enable_moe_block": False,
    }
    actual = {
        key: getattr(text_config, key, None)
        for key in expected
    }
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    architectures = list(getattr(config, "architectures", None) or [])
    core = getattr(base, "model", None)
    encoder = getattr(core, "encoder", None)
    decoder = getattr(core, "decoder", None)
    encoder_text = getattr(encoder, "language_model", None)
    tied_embeddings = False
    tied_first_layer = False
    tied_lm_head = False
    comparable_parameter_count = 0
    untied_parameter_names: list[str] = []
    if encoder_text is not None and decoder is not None:
        tied_embeddings = (
            encoder_text.embed_tokens.weight is decoder.embed_tokens.weight
        )
        encoder_gate = encoder_text.layers[0].mlp.gate_proj
        decoder_gate = decoder.layers[0].mlp.gate_proj
        decoder_base = getattr(decoder_gate, "base_layer", decoder_gate)
        tied_first_layer = encoder_gate.weight is decoder_base.weight
        tied_lm_head = (
            getattr(base, "lm_head", None) is not None
            and base.lm_head.weight is decoder.embed_tokens.weight
        )
        for name, encoder_parameter in encoder_text.named_parameters():
            module_path, _, parameter_name = name.rpartition(".")
            try:
                decoder_module = (
                    decoder.get_submodule(module_path)
                    if module_path
                    else decoder
                )
            except AttributeError:
                continue
            decoder_module = getattr(
                decoder_module, "base_layer", decoder_module
            )
            decoder_parameter = getattr(
                decoder_module, parameter_name, None
            )
            if not isinstance(decoder_parameter, torch.nn.Parameter):
                continue
            comparable_parameter_count += 1
            if encoder_parameter is not decoder_parameter:
                untied_parameter_names.append(name)
    report = {
        "architecture": type(base).__name__,
        "declared_architectures": architectures,
        "text_config": actual,
        "mismatches": mismatches,
        "dense_mlp": getattr(text_config, "enable_moe_block", None) is False,
        "per_layer_embeddings": int(
            getattr(text_config, "hidden_size_per_layer_input", 0) or 0
        )
        > 0,
        "shared_kv_layers": getattr(text_config, "num_kv_shared_layers", None),
        "has_audio_encoder": bool(
            encoder is not None and getattr(encoder, "audio_tower", None) is not None
        ),
        "decoder_class": type(decoder).__name__ if decoder is not None else None,
        "encoder_decoder_embeddings_tied": tied_embeddings,
        "encoder_decoder_first_layer_tied": tied_first_layer,
        "lm_head_decoder_embedding_tied": tied_lm_head,
        "comparable_encoder_decoder_parameters": comparable_parameter_count,
        "untied_encoder_decoder_parameters": untied_parameter_names[:50],
    }
    report["valid"] = all(
        (
            not mismatches,
            architectures == ["MultimodalDiffusionGemmaForBlockDiffusion"],
            report["has_audio_encoder"],
            report["decoder_class"] == "E4BDiffusionDecoderModel",
            tied_embeddings,
            tied_first_layer,
            tied_lm_head,
            comparable_parameter_count > 0,
            not untied_parameter_names,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/corruption"))
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/validation_report.json"))
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    artifact = validate_artifact_files(args.model_dir)
    if artifact.get("exists") and not artifact.get("has_weights"):
        raise RuntimeError(f"artifact has no full model or adapter weights: {args.model_dir}")
    if artifact.get("has_adapter") and not artifact.get("adapter_base_model") and not args.base_model:
        raise RuntimeError("adapter artifact does not declare its base model")

    shard_report = validate_shards(args.data_dir)
    shard_report["fingerprint"] = corruption_data_fingerprint(args.data_dir)
    model, processor, load_metadata = load_model_and_processor(
        args.model_dir,
        processor_source=args.processor,
        base_model=args.base_model,
        dtype=args.dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
    )
    report: dict[str, Any] = {
        "success": True,
        "artifact": artifact,
        "load": load_metadata,
        "processor": validate_processor(processor),
        "data": shard_report,
        "generation_contract": validate_generation_contract(model),
        "e4b_architecture": validate_e4b_architecture(model),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "packages": _package_versions(),
        },
    }
    artifact_fingerprint = load_metadata.get("artifact_data_fingerprint")
    if artifact_fingerprint and artifact_fingerprint != shard_report["fingerprint"]:
        raise RuntimeError(
            "validation corruption data differs from the dataset recorded by the final artifact"
        )
    if not report["processor"]["has_chat_template"]:
        raise RuntimeError("artifact processor/tokenizer has no chat template")
    if not hasattr(processor, "tokenizer"):
        raise RuntimeError(
            "production artifact did not load an AutoProcessor; tokenizer-only fallback "
            "would silently break image/video inputs"
        )
    if not report["generation_contract"]["strict_diffusion_not_ar_fallback"]:
        raise RuntimeError(
            "loaded artifact does not satisfy the strict DiffusionGemma generation contract: "
            f"{report['generation_contract']}"
        )
    if not report["e4b_architecture"]["valid"]:
        raise RuntimeError(
            "loaded artifact is not the required E4B Diffusion Transformer: "
            f"{report['e4b_architecture']}"
        )
    if not args.skip_forward:
        report["forward"] = validate_forward(model, args.data_dir)
        if not report["forward"]["finite_loss"]:
            raise RuntimeError("masked denoising forward loss is not finite")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
