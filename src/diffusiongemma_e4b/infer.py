from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoProcessor, AutoTokenizer

from .constants import CANVAS_LENGTH, DEFAULT_STUDENT_MODEL


MULTIMODAL_BATCH_KEYS = (
    "pixel_values",
    "input_features",
    "input_features_mask",
    "image_position_ids",
    "mm_token_type_ids",
)


def entropy_from_logits(logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = logits.float() / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    log_probs = torch.log_softmax(scaled, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return probs, entropy


def _joint_entropy_accept_mask(entropy: torch.Tensor, entropy_bound: float) -> torch.Tensor:
    """Select the largest lowest-entropy set whose joint bound is satisfied.

    This mirrors the EntropyBoundSampler rule used by DiffusionGemma: for sorted
    entropies h_1..h_k, select the largest k for which sum(h_1..h_k)-h_k is no
    greater than the configured bound. A single position is therefore always
    accepted.
    """

    if entropy.ndim != 2:
        raise ValueError(f"expected [batch, sequence] entropy, got {tuple(entropy.shape)}")
    accept = torch.zeros_like(entropy, dtype=torch.bool)
    for batch_idx in range(entropy.shape[0]):
        sorted_entropy, sorted_index = torch.sort(entropy[batch_idx])
        admissible = torch.cumsum(sorted_entropy, dim=0) - sorted_entropy <= float(entropy_bound)
        count = int(admissible.long().sum().item())
        count = max(1, min(count, entropy.shape[1]))
        accept[batch_idx, sorted_index[:count]] = True
    return accept


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
    accept = _joint_entropy_accept_mask(entropy, entropy_bound)
    random_tokens = torch.randint(
        0,
        vocab_size,
        current_canvas.shape,
        device=current_canvas.device,
        generator=generator,
    )
    next_canvas = torch.where(accept, candidate, random_tokens)
    return next_canvas, candidate, entropy


def _model_input_device(model) -> torch.device:
    try:
        device = model.get_input_embeddings().weight.device
        if device.type != "meta":
            return device
    except Exception:  # noqa: BLE001
        pass
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("could not determine the model input device")


def _as_device_batch(inputs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
    return result


def _prompt_inputs(processor, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if hasattr(processor, "apply_chat_template"):
        try:
            encoded = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if isinstance(encoded, torch.Tensor):
                encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
            return _as_device_batch(dict(encoded), device)
        except (TypeError, ValueError, KeyError):
            # Older tokenizer-only processors may require string chat content.
            text_messages = [{"role": "user", "content": prompt}]
            try:
                encoded = processor.apply_chat_template(
                    text_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if isinstance(encoded, torch.Tensor):
                    encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
                return _as_device_batch(dict(encoded), device)
            except (TypeError, ValueError, KeyError):
                pass

    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    return _as_device_batch(dict(encoded), device)


def _set_entropy_bound(generation_config, entropy_bound: float) -> None:
    sampler = getattr(generation_config, "sampler_config", None)
    if sampler is None:
        try:
            sampler_config_cls = getattr(transformers, "EntropyBoundSamplerConfig")
            generation_config.sampler_config = sampler_config_cls(entropy_bound=entropy_bound)
            return
        except (AttributeError, TypeError):
            generation_config.sampler_config = {
                "_cls_name": "EntropyBoundSamplerConfig",
                "entropy_bound": entropy_bound,
            }
            return
    if isinstance(sampler, dict):
        sampler["entropy_bound"] = entropy_bound
    else:
        setattr(sampler, "entropy_bound", entropy_bound)


@torch.no_grad()
def _transformers_diffusion_generate(
    model,
    processor,
    prompt: str,
    max_new_tokens: int,
    denoise_steps: int,
    entropy_bound: float,
    confidence_threshold: float,
    stability_steps: int,
    t_min: float,
    t_max: float,
    encoder_inputs: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    device = _model_input_device(model)
    inputs = _as_device_batch(dict(encoder_inputs or {}), device)
    if "input_ids" not in inputs:
        inputs = _prompt_inputs(processor, prompt, device)

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = int(max_new_tokens)
    generation_config.max_denoising_steps = int(denoise_steps)
    generation_config.confidence_threshold = float(confidence_threshold)
    generation_config.stability_threshold = int(stability_steps)
    generation_config.t_min = float(t_min)
    generation_config.t_max = float(t_max)
    _set_entropy_bound(generation_config, entropy_bound)

    outputs = model.generate(**inputs, generation_config=generation_config)
    sequences = getattr(outputs, "sequences", outputs)
    if not isinstance(sequences, torch.Tensor):
        raise TypeError(f"model.generate returned unsupported type: {type(outputs).__name__}")
    prompt_length = int(inputs["input_ids"].shape[-1])
    new_ids = sequences[:, prompt_length:] if sequences.shape[-1] >= prompt_length else sequences
    text = processor.decode(new_ids[0], skip_special_tokens=True)
    return {
        "text": text,
        "blocks": max(1, math.ceil(max_new_tokens / CANVAS_LENGTH)),
        "strict_diffusion": True,
        "ar_fallback_used": False,
        "sampler": "transformers.EntropyBoundSampler",
        "generation": {
            "max_denoising_steps": denoise_steps,
            "entropy_bound": entropy_bound,
            "confidence_threshold": confidence_threshold,
            "stability_threshold": stability_steps,
            "t_min": t_min,
            "t_max": t_max,
        },
    }


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
    encoder_inputs: dict[str, torch.Tensor] | None = None,
    t_min: float = 0.4,
    t_max: float = 0.8,
) -> dict[str, Any]:
    """Generate with the Transformers DiffusionGemma mixin when available.

    A small manual implementation is kept only for unit-test doubles. Production
    artifacts must expose the real ``generate`` method and generation config.
    """

    if callable(getattr(model, "generate", None)) and getattr(model, "generation_config", None) is not None:
        model_canvas_length = int(getattr(getattr(model, "config", None), "canvas_length", canvas_length))
        if canvas_length != model_canvas_length:
            raise ValueError(
                f"requested canvas_length={canvas_length}, but the loaded DiffusionGemma "
                f"checkpoint requires canvas_length={model_canvas_length}"
            )
        return _transformers_diffusion_generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            denoise_steps,
            entropy_bound,
            confidence_threshold,
            stability_steps,
            t_min,
            t_max,
            encoder_inputs,
        )

    device = _model_input_device(model)
    encoder_inputs = dict(encoder_inputs or {})
    base_attention_mask = None
    if "input_ids" in encoder_inputs:
        input_ids = encoder_inputs.pop("input_ids").to(device)
        base_attention_mask = encoder_inputs.pop("attention_mask", None)
        if base_attention_mask is not None:
            base_attention_mask = base_attention_mask.to(device)
    else:
        encoded = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        input_ids = encoded.to(device)
    if base_attention_mask is None:
        base_attention_mask = torch.ones_like(input_ids)
    encoder_inputs = {k: v.to(device) for k, v in encoder_inputs.items()}
    committed = input_ids
    committed_attention_mask = base_attention_mask
    blocks: list[dict[str, Any]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    vocab_size = int(model.config.text_config.vocab_size)
    eos_ids = getattr(tokenizer, "eos_token_id", None)
    eos_ids = {int(eos_ids)} if isinstance(eos_ids, int) else set(eos_ids or [])

    remaining = max_new_tokens
    stop_all = False
    while remaining > 0 and not stop_all:
        block_size = min(canvas_length, remaining)
        canvas = torch.randint(0, vocab_size, (1, canvas_length), device=device, generator=generator)
        self_conditioning_logits = None
        encoder_past_key_values = None
        last_argmax = None
        stable = 0
        trace: list[dict[str, Any]] = []
        final_candidate = canvas
        for step in range(denoise_steps):
            fraction = step / max(denoise_steps - 1, 1)
            step_temperature = float(t_max + fraction * (t_min - t_max))
            model_kwargs: dict[str, Any] = {
                "decoder_input_ids": canvas,
                "decoder_attention_mask": torch.cat(
                    [committed_attention_mask, torch.ones_like(canvas)], dim=1
                ),
                "self_conditioning_logits": self_conditioning_logits,
                **encoder_inputs,
            }
            if encoder_past_key_values is None:
                model_kwargs["input_ids"] = committed
                model_kwargs["attention_mask"] = committed_attention_mask
            else:
                model_kwargs["past_key_values"] = encoder_past_key_values
            out = model(**model_kwargs)
            logits = out.logits
            encoder_past_key_values = out.past_key_values
            canvas, argmax_canvas, entropy = entropy_bound_step(
                logits,
                canvas,
                entropy_bound,
                step_temperature if temperature == 1.0 else temperature,
                vocab_size,
                generator,
            )
            final_candidate = argmax_canvas
            self_conditioning_logits = logits.detach()
            mean_entropy = float(entropy.mean().detach().cpu())
            if last_argmax is not None and torch.equal(argmax_canvas, last_argmax):
                stable += 1
            else:
                stable = 0
            last_argmax = argmax_canvas.detach().clone()
            trace.append(
                {
                    "step": step + 1,
                    "mean_entropy": mean_entropy,
                    "stable": stable,
                    "temperature": step_temperature,
                }
            )
            if mean_entropy <= confidence_threshold and stable >= stability_steps:
                break

        commit = final_candidate[:, :block_size]
        if eos_ids:
            eos_positions = [idx for idx, token in enumerate(commit[0].tolist()) if int(token) in eos_ids]
            if eos_positions:
                commit = commit[:, : eos_positions[0] + 1]
                stop_all = True
        committed = torch.cat([committed, commit], dim=1)
        committed_attention_mask = torch.cat(
            [
                committed_attention_mask,
                torch.ones(
                    (committed_attention_mask.shape[0], commit.shape[1]),
                    dtype=committed_attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        blocks.append({"tokens": int(commit.numel()), "steps": len(trace), "trace": trace})
        remaining -= int(commit.numel())

    new_ids = committed[:, input_ids.shape[1] :]
    return {
        "text": tokenizer.decode(new_ids[0], skip_special_tokens=True),
        "blocks": blocks,
        "strict_diffusion": True,
        "ar_fallback_used": False,
        "sampler": "legacy-test-only-joint-entropy-bound",
        "canvas_length": canvas_length,
    }


def _load_base_model(
    source: str,
    dtype: torch.dtype,
    device_map: str,
    load_in_4bit: bool,
    revision: str | None = None,
):
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if revision and not Path(source).exists():
        kwargs["revision"] = revision
    if load_in_4bit:
        quantization_cls = getattr(transformers, "BitsAndBytesConfig", None)
        if quantization_cls is None:
            raise RuntimeError("--load-in-4bit requires a Transformers build with BitsAndBytesConfig")
        kwargs["quantization_config"] = quantization_cls(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    config = transformers.AutoConfig.from_pretrained(
        source,
        trust_remote_code=True,
        revision=revision,
    )
    architectures = set(getattr(config, "architectures", None) or [])
    if "MultimodalDiffusionGemmaForBlockDiffusion" in architectures:
        from .modeling_multimodal import MultimodalDiffusionGemmaForBlockDiffusion

        return MultimodalDiffusionGemmaForBlockDiffusion.from_pretrained(source, **kwargs)
    raise RuntimeError(
        f"{source} does not declare this project's E4B diffusion architecture; "
        "pass the transplanted E4B base instead"
    )


def _find_bundled_base_model(adapter_path: Path) -> Path | None:
    """Return the self-contained release base next to artifacts/final."""

    candidate = adapter_path.parent / "base_model"
    config_path = candidate / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if config.get("architectures") != [
        "MultimodalDiffusionGemmaForBlockDiffusion"
    ]:
        return None
    return candidate.resolve()


def load_model_and_processor(
    model_source: str | Path,
    processor_source: str | Path | None = None,
    base_model: str | None = None,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    load_in_4bit: bool = False,
):
    source = str(model_source)
    source_path = Path(source)
    adapter_config_path = source_path / "adapter_config.json"
    adapter_config: dict[str, Any] = {}
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    metadata_path = source_path / "artifact_metadata.json"
    if not metadata_path.exists():
        metadata_path = source_path / "training_metadata.json"
    artifact_metadata: dict[str, Any] = {}
    if metadata_path.exists():
        artifact_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    resolved_base = (
        base_model
        or adapter_config.get("base_model_name_or_path")
        or artifact_metadata.get("base_model")
        or artifact_metadata.get("base_model_name_or_path")
    )
    resolved_revision = (
        None
        if base_model
        else artifact_metadata.get("base_model_revision")
        or adapter_config.get("revision")
    )
    bundled_base = (
        _find_bundled_base_model(source_path)
        if adapter_config and base_model is None
        else None
    )
    if bundled_base is not None:
        resolved_base = str(bundled_base)
        resolved_revision = None
    torch_dtype = getattr(torch, dtype)
    if adapter_config:
        if not resolved_base:
            raise RuntimeError("Adapter artifact is missing base_model_name_or_path; pass --base-model explicitly")
        model = _load_base_model(
            str(resolved_base),
            torch_dtype,
            device_map,
            load_in_4bit,
            revision=str(resolved_revision) if resolved_revision else None,
        )
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading this adapter requires PEFT") from exc
        model = PeftModel.from_pretrained(model, source, is_trainable=False)
    else:
        model = _load_base_model(
            source,
            torch_dtype,
            device_map,
            load_in_4bit,
            revision=str(resolved_revision) if resolved_revision else None,
        )

    local_processor = source_path if source_path.exists() and any(
        (source_path / name).exists()
        for name in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json")
    ) else None
    resolved_processor = str(processor_source or local_processor or resolved_base or source)
    try:
        processor = AutoProcessor.from_pretrained(resolved_processor, trust_remote_code=True)
    except Exception:  # noqa: BLE001
        processor = AutoTokenizer.from_pretrained(resolved_processor, trust_remote_code=True)
    model.eval()
    return model, processor, {
        "model_source": source,
        "base_model": resolved_base,
        "base_model_revision": resolved_revision,
        "processor_source": resolved_processor,
        "is_adapter": bool(adapter_config),
        "model_class": type(model).__name__,
        "model_type": getattr(getattr(model, "config", None), "model_type", None),
        "artifact_data_fingerprint": artifact_metadata.get("data_fingerprint"),
        "artifact_training_fingerprint": artifact_metadata.get("training_fingerprint"),
        "best_optimizer_steps": artifact_metadata.get("best_optimizer_steps"),
        "baseline_val_loss": artifact_metadata.get("baseline_val_loss"),
        "best_val_loss": artifact_metadata.get("best_val_loss"),
        "relative_val_improvement": artifact_metadata.get(
            "relative_val_improvement"
        ),
        "minimum_relative_val_improvement": artifact_metadata.get(
            "minimum_relative_val_improvement"
        ),
    }


def load_encoder_inputs(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    import numpy as np

    tensors: dict[str, torch.Tensor] = {}
    with np.load(path) as data:
        for key in ("input_ids", "attention_mask", *MULTIMODAL_BATCH_KEYS):
            if key in data:
                tensors[key] = torch.from_numpy(data[key])
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--prompt", default="Explain block diffusion in one concise paragraph.")
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/strict_diffusion_inference.json"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--denoise-steps", type=int, default=48)
    parser.add_argument("--entropy-bound", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.005)
    parser.add_argument("--stability-steps", type=int, default=1)
    parser.add_argument("--t-min", type=float, default=0.4)
    parser.add_argument("--t-max", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0, help="Legacy sampler only")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--encoder-input-npz", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, processor, load_metadata = load_model_and_processor(
        args.model_dir,
        processor_source=args.processor,
        base_model=args.base_model,
        dtype=args.dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
    )
    result = strict_diffusion_generate(
        model,
        processor,
        args.prompt,
        args.max_new_tokens,
        args.canvas_length,
        args.denoise_steps,
        args.entropy_bound,
        args.confidence_threshold,
        args.stability_steps,
        args.temperature,
        args.seed,
        load_encoder_inputs(args.encoder_input_npz),
        t_min=args.t_min,
        t_max=args.t_max,
    )
    result["load"] = load_metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "blocks": result["blocks"]}, indent=2))


if __name__ == "__main__":
    main()
