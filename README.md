# DiffusionGemma-E4B

This repository contains an executable conversion pipeline for building a Gemma 4 E4B initialized DiffusionGemma-style block diffusion student.

It is not a runtime wrapper. The student architecture is `transformers.DiffusionGemmaForBlockDiffusion` configured with Gemma 4 E4B text dimensions, 256-token canvas denoising, bidirectional decoder attention, self-conditioning inputs, and strict diffusion inference.

Formal training data policy:

- No external datasets.
- No Hugging Face datasets.
- No curated prompt banks.
- No hand-written training prompts.
- Formal records are prompt-free self-continuation text streams produced by Gemma 4 E4B itself.

Key commands are in `scripts/`. The full RunPod handoff is in `docs/RUNPOD_HANDOFF.md`.
