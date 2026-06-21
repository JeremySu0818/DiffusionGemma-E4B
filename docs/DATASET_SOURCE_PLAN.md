# Dataset Source Plan

Purpose: choose data sources for converting Gemma 4 E4B into a DiffusionGemma-style block diffusion model while preserving the original model behavior as much as practical.

This is not an instruction-tuning plan. For conversion, the main artifact should be token streams that teach denoising over the target text distribution. Prompt/context datasets are used mainly to elicit broad Gemma 4 E4B teacher outputs, not as answer labels.

## Recommended Mix

| Stage | Share | Source type | Role |
| --- | ---: | --- | --- |
| Teacher behavior stream | 70% | Broad prompts/context -> Gemma 4 E4B outputs | Main conversion target, keeps student close to teacher behavior. |
| Denoising warm-up stream | 20% | High-quality plain text | Optional warm-up for block denoising stability. Keep small to avoid style drift. |
| Prompt-free self-continuation | 5% | Empty/minimal context teacher generation | Keeps repo's original prompt-free path represented, but not enough diversity alone. |
| Evaluation-only prompts | 5% | Bench/eval prompts | Never train answers from these; use to compare student against teacher. |

If compute is tight, skip the denoising warm-up and spend the budget on teacher behavior stream.

## Green Sources

Use these first.

| Dataset | Hugging Face / Source | Use | Why |
| --- | --- | --- | --- |
| LMSYS-Chat-1M | `lmsys/lmsys-chat-1m` | Prompt/context bank | Large real-world chat prompts across many tasks and languages. Use user turns/context only; discard assistant answers. Requires accepting dataset terms on HF. |
| WildChat-4.8M non-toxic | `allenai/WildChat-4.8M` | Prompt/context bank | Large non-toxic real user prompts. Use user inputs only; dedupe and privacy-filter before teacher generation. |
| WildChat-nontoxic | `allenai/WildChat-nontoxic` | Prompt/context bank | Smaller, cleaner fallback if 4.8M is too large. |
| UltraChat 200k | `HuggingFaceH4/ultrachat_200k` | Prompt bank | Heavily filtered, broad instruction/dialogue prompts. Use prompts only unless explicitly doing style experiments. |
| Tulu 3 SFT Mixture | `allenai/tulu-3-sft-mixture` | Prompt bank / coverage balancer | Useful for math, code, instruction-following, multilingual, safety coverage. Treat subset licenses carefully; use prompts only. |
| 10k Prompts Ranked | `data-is-better-together/10k_prompts_ranked` | Pilot prompt bank | Small clean pilot set for local pipeline testing before large generation. |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | Optional denoising warm-up text | High-quality educational web text. Use text directly only for a short warm-up stage. |
| Cosmopedia | `HuggingFaceTB/cosmopedia` | Optional denoising warm-up text | Synthetic textbook/blog/story/wikiHow style text, Apache-2.0. Good for structured expository language. |
| Wikipedia | `wikimedia/wikipedia`, `20231101.en` | Optional denoising warm-up text | Stable encyclopedic language. Good low-risk baseline text stream, with attribution/license obligations. |
| The Stack v2 | `bigcode/the-stack-v2` | Optional code stream | Use if code behavior matters. Preserve provenance and license metadata. |
| Proof-Pile-2 | `EleutherAI/proof-pile-2` | Optional math/science stream | Use if math/science denoising coverage matters. Keep small unless math behavior is a priority. |

## Amber Sources

Use only with an explicit reason.

| Dataset | Use | Concern |
| --- | --- | --- |
| OpenHermes 2.5 | Prompt bank only | Large and diverse, but mixed/unclear component licensing. Do not use answers as targets. |
| RedPajama-Data-1T / RedPajama-V2 | Plain text warm-up | Massive and useful, but heavier to filter than FineWeb-Edu/SlimPajama-style subsets. Prefer smaller curated slices first. |
| FineWeb full | Plain text warm-up | Excellent scale, but too large for first-pass conversion. Prefer FineWeb-Edu for quality and practicality. |
| SlimPajama | Plain text warm-up | Good deduped pretraining text, but older than FineWeb-Edu and still huge. Use if FineWeb-Edu access is inconvenient. |

## Red Sources

Avoid for the main conversion target.

| Dataset type | Why |
| --- | --- |
| SFT answers from other models | They pull the student toward another assistant's behavior rather than Gemma 4 E4B. |
| Narrow prompt banks only, such as pure coding or pure math | They preserve only a narrow slice of teacher behavior. |
| Raw empty-prompt self-continuation as the main data | It over-samples greetings, empty outputs, and generic assistant startup text. |
| Unfiltered web crawl dumps | Too much cleanup burden for a conversion run; noisy targets make denoising look easier but behavior worse. |

## Prompt/Context Extraction Rules

1. Keep user/system/context text as prompts for Gemma 4 E4B.
2. Do not train on existing assistant answers from prompt datasets.
3. Generate fresh teacher outputs with the target base model.
4. Drop empty prompts, duplicate prompts, personally identifying snippets, toxic prompts unless intentionally building an eval bucket, and prompts that are mostly markup/noise.
5. Preserve language/task/domain tags where available so sampling can be balanced.

## Teacher Stream Filters

Drop generated samples when:

1. `text` is empty or under 32 estimated tokens.
2. It is near-duplicate of another sample.
3. It is mostly greeting or refusal boilerplate.
4. It contains obvious endpoint failure text.
5. It exceeds policy/safety boundaries for the intended release.

## First-Pass Sampling Recipe

Target for a first cloud run:

| Bucket | Share | Candidate sources |
| --- | ---: | --- |
| General chat / QA | 20% | LMSYS-Chat-1M, WildChat non-toxic |
| Reasoning / math | 15% | Tulu 3 math/persona subsets, Proof-Pile-2 prompts if converted |
| Coding | 15% | Tulu 3 Persona Python, The Stack v2 snippets as context |
| Writing / editing / translation | 15% | UltraChat 200k, LMSYS-Chat-1M |
| Knowledge explanation | 15% | FineWeb-Edu contexts, Wikipedia contexts, Cosmopedia contexts |
| Long-form continuation | 10% | FineWeb-Edu, Wikipedia, Cosmopedia |
| Structured output | 5% | Tulu 3 IF, UltraChat prompts |
| Safety / uncertainty eval | 5% | Tulu WildGuard/WildJailbreak for eval and constrained teacher sampling |

The first-pass output should still be Gemma-generated text, then tokenized into 256-token corruption blocks.

## Decision

For this project, the main dataset family should be prompt/context banks plus Gemma 4 E4B generated outputs. Plain text corpora are useful, but only as optional denoising warm-up or context seeds. Do not replace teacher-generated stream with professional SFT answers if the goal is maximum behavior preservation.
