from __future__ import annotations

import argparse
import json
import shlex


# These are defaults, not assignments. Rendered shell snippets only fill
# variables that the caller has not already set.
PRESETS: dict[str, dict[str, object]] = {
    "gpu": {
        "student_init": "transplant",
        "student_model": "google/gemma-4-E4B-it",
        # One full pass over E4B teacher canvases. The custom student preserves
        # E4B weights and adapts only diffusion-decoder behavior.
        "target_estimated_tokens": 60_000_000,
        "target_blocks": 220_000,
        "prefix_length": 2_048,
        "batch_size": 1,
        "grad_accum": 16,
        "max_optimizer_steps": 13_000,
        "lr": "1e-4",
        "warmup_steps": 390,
        "weight_decay": "0.1",
        "lora_r": 64,
        "lora_alpha": 128,
        # Generic Transformers checkpointing drops the encoder KV cache that the
        # diffusion decoder must attend to.
        "gradient_checkpointing": 0,
        "max_shard_uncompressed_bytes": 536_870_912,
        "save_interval": 1_000,
        "val_interval": 500,
        "val_batches": 128,
        "min_relative_val_improvement": "0.005",
        "checkpoint_retention": 3,
        "train_mode_windows": "lora",
        "train_mode_linux": "qlora",
    },
    "large": {
        "student_init": "transplant",
        "student_model": "google/gemma-4-E4B-it",
        "target_estimated_tokens": 550_000_000,
        "target_blocks": 2_000_000,
        "prefix_length": 2_048,
        "batch_size": 1,
        "grad_accum": 32,
        "max_optimizer_steps": 125_000,
        "lr": "5e-5",
        "warmup_steps": 3_750,
        "weight_decay": "0.1",
        "lora_r": 64,
        "lora_alpha": 128,
        "gradient_checkpointing": 0,
        "max_shard_uncompressed_bytes": 536_870_912,
        "save_interval": 1_000,
        "val_interval": 500,
        "val_batches": 128,
        "min_relative_val_improvement": "0.005",
        "checkpoint_retention": 3,
        # Keep a large run independent from the default production artifacts so
        # exact-resume fingerprints never collide across presets.
        "log_dir": "outputs/logs/large",
        "teacher_output": "data/teacher_supervised/large/teacher_outputs.jsonl",
        "teacher_progress": "data/teacher_supervised/large/progress.json",
        "corruption_dir": "data/corruption-large",
        "train_output_dir": "artifacts/conversion_training-large",
        "validation_dir": "outputs/validation-large",
        "export_output": "artifacts/diffusiongemma-e4b-large-bundle.tar.gz",
        "train_mode_windows": "lora",
        "train_mode_linux": "qlora",
    },
    "smoke": {
        "student_init": "transplant",
        "student_model": "google/gemma-4-E4B-it",
        "target_estimated_tokens": 32_768,
        "target_blocks": 128,
        "prefix_length": 512,
        "max_shard_uncompressed_bytes": 536_870_912,
        "batch_size": 1,
        "grad_accum": 1,
        "max_optimizer_steps": 2,
        "lr": "5e-5",
        "warmup_steps": 0,
        "weight_decay": "0.1",
        "gradient_checkpointing": 0,
        "save_interval": 1,
        "val_interval": 1,
        "val_batches": 1,
        "min_relative_val_improvement": "0.0",
        "checkpoint_retention": 1,
        "max_records_per_source": 32,
        "max_total_prompt_records": 256,
        # Smoke must be safe to run immediately before the default GPU preset.
        # Isolate all run-scoped outputs while still sharing immutable HF/media
        # caches.
        "log_dir": "outputs/logs/smoke",
        "teacher_output": "data/teacher_supervised/smoke/teacher_outputs.jsonl",
        "teacher_progress": "data/teacher_supervised/smoke/progress.json",
        "corruption_dir": "data/corruption-smoke",
        "train_output_dir": "artifacts/conversion_training-smoke",
        "validation_dir": "outputs/validation-smoke",
        "export_output": "artifacts/diffusiongemma-e4b-smoke-bundle.tar.gz",
        "train_mode_windows": "lora",
        "train_mode_linux": "qlora",
    },
}


def _env_key(key: str) -> str:
    return "DG_" + key.upper()


def render_bash(values: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in values.items():
        env_key = _env_key(key)
        quoted = shlex.quote(str(value))
        lines.append(f'if [[ -z "${{{env_key}+x}}" ]]; then export {env_key}={quoted}; fi')
    lines.append(
        'if [[ -z "${DG_TRAIN_MODE+x}" ]]; then '
        'export DG_TRAIN_MODE="$DG_TRAIN_MODE_LINUX"; fi'
    )
    return "\n".join(lines)


def render_powershell(values: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in values.items():
        env_key = _env_key(key)
        escaped = str(value).replace("'", "''")
        lines.append(f"if ($null -eq $env:{env_key}) {{ $env:{env_key}='{escaped}' }}")
    lines.append(
        "if ($null -eq $env:DG_TRAIN_MODE) "
        "{ $env:DG_TRAIN_MODE=$env:DG_TRAIN_MODE_WINDOWS }"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gpu")
    parser.add_argument("--shell", choices=["powershell", "bash", "json"], default="json")
    args = parser.parse_args()

    values = PRESETS[args.preset]
    if args.shell == "json":
        print(json.dumps(values, indent=2))
    elif args.shell == "powershell":
        print(render_powershell(values))
    else:
        print(render_bash(values))


if __name__ == "__main__":
    main()
