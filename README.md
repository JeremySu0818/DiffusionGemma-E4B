# DiffusionGemma-E4B

This repository contains an executable conversion pipeline for building a Gemma 4 E4B initialized DiffusionGemma-style block diffusion student.

It is not a runtime wrapper. The student architecture is `transformers.DiffusionGemmaForBlockDiffusion` configured with Gemma 4 E4B text dimensions, 256-token canvas denoising, bidirectional decoder attention, self-conditioning inputs, and strict diffusion inference.

Formal training data policy:

- No external datasets.
- No Hugging Face datasets.
- No curated prompt banks.
- No hand-written training prompts.
- Formal records are prompt-free self-continuation text streams produced by Gemma 4 E4B itself.

Key commands are organized in [scripts/linux/](file:///mnt/c/AI/DiffusionGemma-E4B/scripts/linux) and [scripts/windows/](file:///mnt/c/AI/DiffusionGemma-E4B/scripts/windows). The full deployment guide is in [DEPLOYMENT_GUIDE.md](file:///mnt/c/AI/DiffusionGemma-E4B/docs/DEPLOYMENT_GUIDE.md).

