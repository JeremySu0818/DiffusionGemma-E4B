# Final Report

Execution date: 2026-06-19 Asia/Taipei.

## Delivered In This Folder

- `src/diffusiongemma_e4b/`: executable Python package for config conversion, teacher generation, corruption shards, weight transplant, conversion training, strict diffusion inference, validation, export, and local feasibility probing.
- `configs/diffusiongemma-e4b/config.json`: Gemma 4 E4B-derived DiffusionGemma config, canvas length 256.
- `artifacts/tokenizer_processor_gemma4_e4b/`: Gemma 4 E4B tokenizer, processor, special-token, and chat-template compatibility files saved locally.
- `docs/RUNPOD_HANDOFF.md`: command-level RunPod handoff.
- `docs/DATASET_SOURCE_PLAN.md`: dataset source plan for teacher-stream generation and optional denoising warm-up.
- Validation-only data, local probe outputs, and repro bundles were generated during the local smoke run but are no longer retained in the repository after cleanup.

## Local Runtime Facts Verified

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB VRAM.
- Python: 3.11.9.
- Torch: 2.12.1+cu130, CUDA available.
- LM Studio CLI available and server running on port 1234.
- LM Studio contains `google/gemma-4-e4b`; it was loaded successfully in 2m 11.55s and reported 5.89 GiB.
- After validation, `google/gemma-4-e4b` was unloaded from LM Studio; no long-running local training/generation process was left behind.
- Ollama is installed but does not currently contain Gemma 4 E4B.
- HF CLI is not logged in, so original HF weight download/transplant cannot be completed locally without login/license state.

## Architecture Outcome

The student uses HF `DiffusionGemmaForBlockDiffusion` with Gemma 4 E4B text dimensions:

- vocab size: 262,144
- hidden size: 2,560
- layers: 42
- attention heads: 8
- KV heads: 2
- full/sliding layer pattern preserved
- 256-token canvas
- bidirectional diffusion decoder mode
- causal prefix/encoder mode
- self-conditioning input path

HF DiffusionGemma assumes an MoE branch. Gemma 4 E4B is dense, so the conversion config represents the expert branch as a deterministic single expert:

- `num_experts=1`
- `top_k_experts=1`
- `moe_intermediate_size=10240`
- `num_global_key_value_heads=2`

Meta-device initialization succeeded:

- class: `DiffusionGemmaForBlockDiffusion`
- named parameters: 16,723,096,148
- validation artifact was generated during the smoke run and later removed from the repository cleanup.

## Local Teacher Generation Validation

The local LM Studio OpenAI-compatible route was tested with Gemma 4 E4B:

- primary `/v1/completions` empty prompt was rejected by LM Studio for this chat model
- adapter automatically switched to `/v1/chat/completions` with an empty legal chat skeleton
- no hand-written user/assistant content was supplied
- records generated: 12
- estimated tokens: 1,257
- validation artifact was generated during the smoke run and later removed from the repository cleanup.

This is validation-only and is not claimed as formal training data.

## Corruption Validation

The validation self-continuation stream was tokenized with the Gemma 4 E4B tokenizer and cut into 256-token target blocks:

- blocks: 4
- canvas length: 256
- validation shard was generated during the smoke run and later removed from the repository cleanup.

This is validation-only and is not claimed as formal 200,000-block data.

## Local Formal Training Decision

Formal local conversion training is not feasible on this machine:

- local VRAM: 6GB
- meta-initialized DiffusionGemma-E4B student: 16.7B named parameters
- BF16 weight floor: 31.15GB
- QLoRA weight floor estimate before activations/optimizer/runtime overhead: 9.03GB
- recommended formal training VRAM: 80GB
- PEFT missing locally
- bitsandbytes missing locally
- HF auth not logged in for original model weight download

Decision from the local feasibility probe:

```json
"decision": "runpod_first_cloud_handoff_required"
```

## Formal Checkpoint Status

No formal trained DiffusionGemma-E4B checkpoint is claimed in this local run.

The folder does not contain a completed formal LoRA/QLoRA/full checkpoint trained on >=50,000,000 self-generated tokens and >=200,000 diffusion target blocks. Creating a fake checkpoint was intentionally not done.

## RunPod First Command

After creating the RunPod pod and placing this project at `/workspace/DiffusionGemma-E4B`, run:

```bash
cd /workspace/DiffusionGemma-E4B && bash scripts/runpod_setup.sh
```

Then follow `docs/RUNPOD_HANDOFF.md` exactly.
