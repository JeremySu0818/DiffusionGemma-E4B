# Research Notes

Sources checked on 2026-06-19 Asia/Taipei:

- Google DeepMind DiffusionGemma page: https://deepmind.google/models/gemma/diffusiongemma/
- Google Developers DiffusionGemma guide: https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- Hugging Face Gemma 4 docs: https://huggingface.co/docs/transformers/model_doc/gemma4
- Hugging Face Gemma 4 E4B model card: https://huggingface.co/google/gemma-4-E4B-it
- Hugging Face DiffusionGemma model card: https://huggingface.co/google/diffusiongemma-26B-A4B-it
- vLLM DiffusionGemma integration: https://vllm.ai/blog/2026-06-10-diffusion-gemma
- DiffuLLaMA / DiffuGPT paper: https://arxiv.org/html/2410.17891v2
- DiffuLLaMA code: https://github.com/HKUNLP/DiffuLLaMA
- Google Hackable Diffusion and Gemma adapter: https://github.com/google/hackable_diffusion and https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion

Engineering conclusions:

- DiffusionGemma uses a Gemma 4 backbone, 256-token block canvas, causal encoder/prefix/commit mode, and bidirectional decoder denoising mode.
- vLLM documents entropy-bound denoising, self-conditioning, convergence/stability checks, and 256-token commit blocks.
- Transformers 5.12.1 in this machine has `transformers.models.diffusion_gemma` with `DiffusionGemmaForBlockDiffusion`.
- Gemma 4 E4B text config is 262,144 vocab, 2560 hidden size, 42 layers, 8 attention heads, 2 KV heads, 512 sliding window, alternating full attention every sixth layer, and 128K context.
- The project therefore builds `DiffusionGemmaConfig` from Gemma 4 E4B `text_config` and uses the official DiffusionGemma HF class rather than a toy model.
- Weight transplant maps Gemma 4 E4B text embeddings, decoder layers, norms, projections, MLP/MoE tensors, RoPE/sliding/full-attention config, and LM head into both DiffusionGemma encoder and decoder paths.
- Formal data is prompt-free self-continuation only: empty completion/BOS-equivalent runtime state, no user-authored content, tokenized into 256-token target blocks, then randomly corrupted for denoising training.
- Strict inference in `src/diffusiongemma_e4b/infer.py` never calls `.generate()` and does not use AR fallback.
