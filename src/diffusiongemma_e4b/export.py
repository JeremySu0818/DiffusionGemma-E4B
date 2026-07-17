from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable


MODEL_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model-*.safetensors",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "pytorch_model.bin",
)
SKIP_FILE_NAMES = {
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng_state.pt",
    "latest_checkpoint.txt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if any(part.startswith("checkpoint-") for part in path.parts):
            continue
        if path.name in SKIP_FILE_NAMES:
            continue
        yield path


def _add_file(tar: tarfile.TarFile, path: Path, arcname: str, manifest: list[dict]) -> None:
    stat = path.stat()
    checksum = sha256_file(path)
    tar.add(path, arcname=arcname, recursive=False)
    manifest.append({"path": arcname, "bytes": stat.st_size, "sha256": checksum})


def add_tree(
    tar: tarfile.TarFile,
    root: Path,
    arc_root: str,
    manifest: list[dict],
    excluded_paths: set[Path] | None = None,
) -> None:
    excluded = {item.resolve() for item in (excluded_paths or set())}
    for path in _iter_files(root):
        if path.resolve() in excluded:
            continue
        arcname = f"{arc_root}/{path.relative_to(root).as_posix()}"
        _add_file(tar, path, arcname, manifest)


def _has_model_weights(model_dir: Path) -> bool:
    return any(any(model_dir.glob(pattern)) for pattern in MODEL_WEIGHT_PATTERNS)


def verify_archive(path: Path, expected_manifest: list[dict]) -> dict:
    expected = {row["path"]: row for row in expected_manifest}
    with tarfile.open(path, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("export archive contains duplicate member names")
        if "MANIFEST.json" not in names:
            raise RuntimeError("export archive is missing MANIFEST.json")
        manifest_member = tar.extractfile("MANIFEST.json")
        if manifest_member is None:
            raise RuntimeError("cannot read MANIFEST.json from export archive")
        archived_manifest = json.loads(manifest_member.read().decode("utf-8"))
        if archived_manifest != expected_manifest:
            raise RuntimeError("archived manifest does not match the export plan")
        payload_names = set(names) - {"MANIFEST.json"}
        if payload_names != set(expected):
            raise RuntimeError("archive payload does not match MANIFEST.json")
        for member in members:
            if member.name == "MANIFEST.json":
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read archived member {member.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            row = expected[member.name]
            if member.size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise RuntimeError(f"archive readback checksum failed for {member.name}")
    return {"verified": True, "files": len(expected_manifest)}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _validate_release_reports(validation_dir: Path) -> dict:
    validation_path = validation_dir / "validation_report.json"
    if not validation_path.is_file():
        raise FileNotFoundError(f"validation report is missing: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("success") is not True:
        raise RuntimeError("validation_report.json does not record success=true")
    generation = validation.get("generation_contract") or {}
    if generation.get("strict_diffusion_not_ar_fallback") is not True:
        raise RuntimeError("validation report did not prove the strict-diffusion contract")
    e4b = validation.get("e4b_architecture") or {}
    if e4b.get("valid") is not True:
        raise RuntimeError("validation report did not prove the E4B architecture contract")
    forward = validation.get("forward") or {}
    if forward.get("forward_ok") is not True or forward.get("finite_loss") is not True:
        raise RuntimeError("validation report lacks a finite denoising forward pass")
    load = validation.get("load") or {}
    relative_improvement = load.get("relative_val_improvement")
    minimum_improvement = load.get("minimum_relative_val_improvement")
    if relative_improvement is None or minimum_improvement is None:
        raise RuntimeError(
            "validation report lacks baseline-relative distillation quality metadata"
        )
    if float(relative_improvement) + 1e-12 < float(minimum_improvement):
        raise RuntimeError(
            "artifact did not satisfy its baseline-relative validation improvement gate"
        )

    inference_path = validation_dir / "strict_diffusion_inference.json"
    if not inference_path.is_file():
        raise FileNotFoundError(f"strict diffusion inference report is missing: {inference_path}")
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    if inference.get("strict_diffusion") is not True or inference.get("ar_fallback_used") is not False:
        raise RuntimeError("inference report used a non-diffusion fallback")
    if not str(inference.get("text") or "").strip():
        raise RuntimeError("inference report contains empty generated text")
    return {"validation": validation, "inference": inference}


def export_bundle(
    output: Path,
    model_dir: Path = Path("artifacts/conversion_training/final"),
    validation_dir: Path = Path("outputs/validation"),
    project_root: Path = Path("."),
    include_code: bool = True,
    base_model_dir: Path | None = None,
) -> dict:
    project_root = project_root.resolve()
    model_dir = model_dir.resolve()
    validation_dir = validation_dir.resolve()
    output = output.resolve()
    if base_model_dir is not None:
        base_model_dir = base_model_dir.resolve()
    summary_path = output.with_suffix(output.suffix + ".json")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"final model directory does not exist: {model_dir}")
    if not _has_model_weights(model_dir):
        raise FileNotFoundError(f"no final model or adapter weights found in {model_dir}")
    is_adapter = (model_dir / "adapter_config.json").is_file()
    if is_adapter and base_model_dir is None:
        adapter = json.loads(
            (model_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
        candidate_value = str(adapter.get("base_model_name_or_path") or "").strip()
        if candidate_value:
            candidate = Path(candidate_value)
            if candidate.is_dir():
                base_model_dir = candidate.resolve()
    if is_adapter:
        if base_model_dir is None or not base_model_dir.is_dir():
            raise FileNotFoundError(
                "the E4B adapter requires its transplanted base model in the release bundle"
            )
        if not _has_model_weights(base_model_dir):
            raise FileNotFoundError(
                f"no transplanted E4B base weights found in {base_model_dir}"
            )
        base_config = json.loads(
            (base_model_dir / "config.json").read_text(encoding="utf-8")
        )
        if base_config.get("architectures") != [
            "MultimodalDiffusionGemmaForBlockDiffusion"
        ]:
            raise RuntimeError("release base model is not the custom E4B architecture")
    if not validation_dir.is_dir():
        raise FileNotFoundError(f"validation directory is missing: {validation_dir}")
    release_reports = _validate_release_reports(validation_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    temporary_handle = tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        # Safetensors are only weakly compressible. Level 1 avoids spending
        # hours of cloud CPU for negligible release-size savings.
        with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
            excluded = {output, summary_path, temporary}
            add_tree(tar, model_dir, "artifacts/final", manifest, excluded_paths=excluded)
            if base_model_dir is not None:
                add_tree(
                    tar,
                    base_model_dir,
                    "artifacts/base_model",
                    manifest,
                    excluded_paths=excluded,
                )
            add_tree(tar, validation_dir, "outputs/validation", manifest, excluded_paths=excluded)
            if include_code:
                for root_name in ("src", "scripts", "configs", "tests", "docs"):
                    root = project_root / root_name
                    add_tree(tar, root, root_name, manifest, excluded_paths=excluded)
                for file_name in ("pyproject.toml", "README.md"):
                    path = project_root / file_name
                    if path.is_file() and not path.is_symlink() and path.resolve() not in excluded:
                        _add_file(tar, path, file_name, manifest)
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            info = tarfile.TarInfo("MANIFEST.json")
            info.size = len(manifest_bytes)
            info.mode = 0o644
            tar.addfile(info, fileobj=io.BytesIO(manifest_bytes))
        verification = verify_archive(temporary, manifest)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    summary = {
        "bundle": str(output),
        "files": len(manifest),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "model_dir": str(model_dir),
        "base_model_dir": str(base_model_dir) if base_model_dir else None,
        "validation_dir": str(validation_dir),
        "strict_diffusion_validated": True,
        "inference_validated": release_reports["inference"] is not None,
        **verification,
    }
    _atomic_write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an atomic, readback-verified release bundle")
    parser.add_argument("--output", type=Path, default=Path("artifacts/diffusiongemma-e4b-repro-bundle.tar.gz"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/conversion_training/final"))
    parser.add_argument("--validation-dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--base-model-dir", type=Path, default=None)
    parser.add_argument("--no-code", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            export_bundle(
                args.output,
                args.model_dir,
                args.validation_dir,
                include_code=not args.no_code,
                base_model_dir=args.base_model_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
