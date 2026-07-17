from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig
from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
    DiffusionGemmaConfig,
    DiffusionGemmaTextConfig,
)

from .constants import CANVAS_LENGTH, DEFAULT_BASE_MODEL


TEXT_KEYS = {
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_activation",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "pad_token_id",
    "eos_token_id",
    "bos_token_id",
    "tie_word_embeddings",
    "rope_parameters",
    "attention_bias",
    "attention_dropout",
    "sliding_window",
    "layer_types",
    "num_global_key_value_heads",
    "global_head_dim",
    "num_experts",
    "top_k_experts",
    "moe_intermediate_size",
    "final_logit_softcapping",
    # Gemma 4 E4B-specific structure. DiffusionGemmaTextConfig preserves these
    # additional fields so the transplanted student keeps the E4B topology.
    "vocab_size_per_layer_input",
    "hidden_size_per_layer_input",
    "num_kv_shared_layers",
    "enable_moe_block",
    "use_double_wide_mlp",
    "attention_k_eq_v",
}


def _clean_text_kwargs(text_dict: dict[str, Any]) -> dict[str, Any]:
    kwargs = {k: text_dict[k] for k in TEXT_KEYS if k in text_dict}
    # The prompt encoder stays causal exactly like Gemma 4 E4B. The custom
    # diffusion decoder makes only the canvas bidirectional.
    kwargs["use_bidirectional_attention"] = text_dict.get(
        "use_bidirectional_attention"
    )
    kwargs.setdefault("final_logit_softcapping", 30.0)
    if kwargs.get("num_global_key_value_heads") is None:
        kwargs["num_global_key_value_heads"] = kwargs["num_key_value_heads"]
    return kwargs


def build_diffusion_e4b_config(
    base_model: str = DEFAULT_BASE_MODEL,
    canvas_length: int = CANVAS_LENGTH,
    dtype: str = "bfloat16",
) -> DiffusionGemmaConfig:
    """Build the E4B-sized diffusion config from Gemma 4 E4B.

    The custom model keeps E4B's dense MLP, per-layer embeddings, shared-KV
    layout and multimodal towers. Only decoder attention semantics and the
    self-conditioning module are diffusion-specific.
    """

    base_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    return build_diffusion_e4b_config_from_base_config(
        base_cfg,
        canvas_length=canvas_length,
        dtype=dtype,
    )


def build_diffusion_e4b_config_from_base_config(
    base_cfg,
    canvas_length: int = CANVAS_LENGTH,
    dtype: str = "bfloat16",
) -> DiffusionGemmaConfig:
    """Build the student config from the exact config used to load E4B weights."""

    if not hasattr(base_cfg, "text_config"):
        raise ValueError("the Gemma 4 E4B base config does not expose text_config")

    text_kwargs = _clean_text_kwargs(base_cfg.text_config.to_dict())
    text_kwargs["dtype"] = dtype
    text_config = DiffusionGemmaTextConfig(**text_kwargs)

    vision_config = None
    if getattr(base_cfg, "vision_config", None) is not None:
        vision_config = base_cfg.vision_config.to_dict()
    audio_config = None
    if getattr(base_cfg, "audio_config", None) is not None:
        audio_config = base_cfg.audio_config.to_dict()

    cfg = DiffusionGemmaConfig(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=audio_config,
        canvas_length=canvas_length,
        dtype=dtype,
        tie_word_embeddings=getattr(base_cfg, "tie_word_embeddings", True),
        boi_token_id=getattr(base_cfg, "boi_token_id", 255999),
        eoi_token_id=getattr(base_cfg, "eoi_token_id", 258882),
        image_token_id=getattr(base_cfg, "image_token_id", 258880),
        boa_token_id=getattr(base_cfg, "boa_token_id", 256000),
        eoa_token_index=getattr(base_cfg, "eoa_token_index", 258883),
        audio_token_id=getattr(base_cfg, "audio_token_id", 258881),
        video_token_id=getattr(base_cfg, "video_token_id", 258884),
    )
    cfg.architectures = ["MultimodalDiffusionGemmaForBlockDiffusion"]
    cfg.name_or_path = "DiffusionGemma-E4B"
    return cfg


def save_config(
    output_dir: Path,
    base_model: str,
    canvas_length: int,
    dtype: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_diffusion_e4b_config(
        base_model,
        canvas_length,
        dtype,
    )
    cfg.save_pretrained(output_dir)
    metadata = {
        "base_model": base_model,
        "canvas_length": canvas_length,
        "dtype": dtype,
        "student_architecture": "diffusiongemma_e4b.modeling_multimodal.MultimodalDiffusionGemmaForBlockDiffusion",
        "conversion": "Gemma 4 E4B weights transplanted into an E4B-sized block-diffusion model",
        "production_supported": True,
        "preserved_e4b_features": [
            "dense_mlp",
            "per_layer_embeddings",
            "shared_kv",
            "vision",
            "audio",
        ],
    }
    (output_dir / "diffusiongemma_e4b_config_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_dir / "config.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/diffusiongemma-e4b"))
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    path = save_config(
        args.output_dir,
        args.base_model,
        args.canvas_length,
        args.dtype,
    )
    print(path)


if __name__ == "__main__":
    main()
