from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from diffusiongemma_e4b.export import export_bundle
from diffusiongemma_e4b.pipeline_presets import PRESETS, render_bash
from diffusiongemma_e4b.preflight import validate_dataset_config, validate_numeric_settings
from diffusiongemma_e4b.train import CorruptionShardDataset


def test_gpu_preset_has_production_defaults() -> None:
    preset = PRESETS["gpu"]
    assert preset["student_init"] == "transplant"
    assert preset["student_model"] == "google/gemma-4-E4B-it"
    assert preset["target_estimated_tokens"] == 60_000_000
    assert preset["target_blocks"] == 220_000
    assert preset["prefix_length"] == 2_048
    assert preset["grad_accum"] == 16
    assert preset["max_optimizer_steps"] == 13_000
    assert preset["lr"] == "1e-4"
    assert preset["warmup_steps"] == 390
    assert preset["lora_r"] == 64
    assert preset["lora_alpha"] == 128
    assert preset["gradient_checkpointing"] == 0
    assert preset["val_batches"] == 128
    assert preset["min_relative_val_improvement"] == "0.005"
    assert preset["checkpoint_retention"] == 3
    assert "train_output_dir" not in preset


def test_large_preset_is_explicit_long_run() -> None:
    preset = PRESETS["large"]
    assert preset["target_estimated_tokens"] == 550_000_000
    assert preset["target_blocks"] == 2_000_000
    assert preset["max_optimizer_steps"] == 125_000
    assert preset["train_output_dir"] == "artifacts/conversion_training-large"


def test_smoke_preset_is_bounded() -> None:
    preset = PRESETS["smoke"]
    assert preset["target_blocks"] == 128
    assert preset["max_optimizer_steps"] == 2
    assert preset["teacher_output"].startswith("data/teacher_supervised/smoke/")
    assert preset["corruption_dir"] == "data/corruption-smoke"
    assert preset["train_output_dir"] == "artifacts/conversion_training-smoke"
    assert preset["validation_dir"] == "outputs/validation-smoke"
    assert preset["export_output"].endswith("-smoke-bundle.tar.gz")


def test_bash_preset_never_overwrites_existing_env() -> None:
    snippet = render_bash(PRESETS["gpu"])
    command = f'export DG_LR=9e-6 DG_TRAIN_MODE=full; {snippet}; printf "%s|%s" "$DG_LR" "$DG_TRAIN_MODE"'
    result = subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
    assert result.stdout == "9e-6|full"


def _numeric_env() -> dict[str, str]:
    return {
        "DG_TARGET_ESTIMATED_TOKENS": "60000000",
        "DG_TARGET_BLOCKS": "220000",
        "DG_CANVAS_LENGTH": "256",
        "DG_PREFIX_LENGTH": "2048",
        "DG_BATCH_SIZE": "1",
        "DG_GRAD_ACCUM": "16",
        "DG_MAX_OPTIMIZER_STEPS": "13000",
        "DG_WARMUP_STEPS": "390",
        "DG_SAVE_INTERVAL": "1000",
        "DG_VAL_INTERVAL": "500",
        "DG_CHECKPOINT_RETENTION": "3",
        "DG_LR": "1e-4",
        "DG_WEIGHT_DECAY": "0.1",
    }


def test_numeric_preflight_accepts_production_relationships() -> None:
    settings, findings = validate_numeric_settings(_numeric_env())
    assert settings["block_token_capacity"] == 56_320_000
    assert settings["planned_block_passes"] == 0.9455
    assert not [item for item in findings if item.level == "error"]


def test_numeric_preflight_rejects_teacher_target_that_cannot_fill_blocks() -> None:
    env = _numeric_env()
    env["DG_TARGET_ESTIMATED_TOKENS"] = "50000000"
    _, findings = validate_numeric_settings(env)
    assert "insufficient_teacher_tokens" in {
        item.code for item in findings if item.level == "error"
    }


def test_dataset_preflight_rejects_unnormalized_or_insufficient_mix(tmp_path: Path) -> None:
    config = {
        "recommended_mix": [{"bucket": "text", "share": 0.8}],
        "sources": [{"id": "tiny", "enabled": True, "max_records": 2}],
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    _, findings = validate_dataset_config(path, {"DG_MAX_TOKENS_PER_SAMPLE": "100"}, target_tokens=1_000)
    codes = {item.code for item in findings if item.level == "error"}
    assert codes == {"dataset_mix_not_normalized", "source_capacity_too_small"}


def test_record_split_is_stratified_by_bucket(tmp_path: Path) -> None:
    record_ids = np.asarray(
        [f"general-{index}" for index in range(99)] + ["image-0", "image-1"]
    )
    buckets = np.asarray(["general"] * 99 + ["image", "image"])
    rows = len(record_ids)
    np.savez_compressed(
        tmp_path / "corruption_text_000000.npz",
        prefix_ids=np.ones((rows, 2), dtype=np.int64),
        prefix_lens=np.full(rows, 2, dtype=np.int64),
        target_ids=np.ones((rows, 2), dtype=np.int64),
        corrupted_ids=np.ones((rows, 2), dtype=np.int64),
        corruption_masks=np.ones((rows, 2), dtype=np.bool_),
        noise_t=np.full(rows, 0.5, dtype=np.float32),
        record_ids=record_ids,
        buckets=buckets,
    )

    train = CorruptionShardDataset(tmp_path, split="train", val_fraction=0.01)
    validation = CorruptionShardDataset(
        tmp_path, split="val", val_fraction=0.01
    )

    assert set(train.bucket_labels) == {"general", "image"}
    assert set(validation.bucket_labels) == {"general", "image"}
    train_rows = {train.index[index][1] for index in range(len(train))}
    validation_rows = {
        validation.index[index][1] for index in range(len(validation))
    }
    assert train_rows.isdisjoint(validation_rows)


def test_export_contains_only_final_validation_and_optional_code(tmp_path: Path) -> None:
    model = tmp_path / "artifacts" / "conversion_training" / "final"
    model.mkdir(parents=True)
    base = tmp_path / "artifacts" / "transplanted"
    base.mkdir(parents=True)
    (base / "model.safetensors").write_bytes(b"transplanted-e4b")
    (base / "config.json").write_text(
        json.dumps(
            {
                "architectures": [
                    "MultimodalDiffusionGemmaForBlockDiffusion"
                ]
            }
        ),
        encoding="utf-8",
    )
    (model / "adapter_model.safetensors").write_bytes(b"adapter")
    (model / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)}),
        encoding="utf-8",
    )
    (model / "optimizer.pt").write_bytes(b"do not export")
    checkpoint = model / "checkpoint-00000001"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"old")
    validation = tmp_path / "outputs" / "validation"
    validation.mkdir(parents=True)
    (validation / "validation_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "generation_contract": {"strict_diffusion_not_ar_fallback": True},
                "e4b_architecture": {"valid": True},
                "forward": {"forward_ok": True, "finite_loss": True},
                "load": {
                    "relative_val_improvement": 0.01,
                    "minimum_relative_val_improvement": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )
    (validation / "strict_diffusion_inference.json").write_text(
        json.dumps(
            {
                "strict_diffusion": True,
                "ar_fallback_used": False,
                "text": "A non-empty diffusion response.",
            }
        ),
        encoding="utf-8",
    )
    data = tmp_path / "data"
    data.mkdir()
    (data / "private.jsonl").write_text("secret", encoding="utf-8")
    output = model / "bundle.tar.gz"
    output.write_bytes(b"old self")

    summary = export_bundle(output, model, validation, project_root=tmp_path, include_code=False)

    assert summary["verified"] is True
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "artifacts/final/adapter_model.safetensors" in names
    assert "artifacts/final/adapter_config.json" in names
    assert "artifacts/base_model/model.safetensors" in names
    assert "artifacts/base_model/config.json" in names
    assert "outputs/validation/validation_report.json" in names
    assert "MANIFEST.json" in names
    assert not any("optimizer" in name or "checkpoint-" in name or "private" in name or "bundle.tar" in name for name in names)
