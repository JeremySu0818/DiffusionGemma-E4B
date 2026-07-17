"""Production tooling for the Gemma 4 E4B block-diffusion conversion pipeline.

The custom E4B architecture is imported lazily so data preparation and teacher
generation remain lightweight.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "build_diffusion_e4b_config",
    "create_diffusion_e4b_model",
    "MultimodalDiffusionGemmaEncoderModel",
    "MultimodalDiffusionGemmaForBlockDiffusion",
    "MultimodalDiffusionGemmaModel",
]


def __getattr__(name: str):
    if name == "build_diffusion_e4b_config":
        return import_module(".config", __name__).build_diffusion_e4b_config
    if name == "create_diffusion_e4b_model":
        return import_module(".student", __name__).create_diffusion_e4b_model
    if name in {
        "MultimodalDiffusionGemmaEncoderModel",
        "MultimodalDiffusionGemmaForBlockDiffusion",
        "MultimodalDiffusionGemmaModel",
    }:
        return getattr(import_module(".modeling_multimodal", __name__), name)
    raise AttributeError(name)
