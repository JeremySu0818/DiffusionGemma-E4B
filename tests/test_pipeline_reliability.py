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




def test_dataset_preflight_rejects_unnormalized_mix(tmp_path: Path) -> None:
    config = {
        "recommended_mix": [{"bucket": "text", "share": 0.8}],
        "sources": [{"id": "tiny", "enabled": True, "max_records": 2}],
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    _, findings = validate_dataset_config(path, {"DG_MAX_TOKENS_PER_SAMPLE": "100"}, target_tokens=1_000)
    codes = {item.code for item in findings if item.level == "error"}
    assert codes == {"dataset_mix_not_normalized"}


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
        vocab_size=np.array([128], dtype=np.int64),
        pad_token_id=np.array([0], dtype=np.int64),
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
