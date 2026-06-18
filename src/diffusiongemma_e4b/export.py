from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_tree(tar: tarfile.TarFile, root: Path, arc_root: str, manifest: list[dict]) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if any(part == "__pycache__" for part in path.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file():
            arcname = f"{arc_root}/{path.relative_to(root).as_posix()}"
            tar.add(path, arcname=arcname)
            manifest.append({"path": arcname, "bytes": path.stat().st_size, "sha256": sha256_file(path)})


def export_bundle(
    output: Path,
    include_code: bool = True,
    artifacts_dir: Path = Path("artifacts"),
    data_dir: Path = Path("data"),
    validation_dir: Path = Path("outputs/validation"),
    logs_dir: Path = Path("outputs/logs"),
    docs_dir: Path = Path("docs"),
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    with tarfile.open(output, "w:gz") as tar:
        if include_code:
            for root_name in ["src", "scripts", "configs", "tests"]:
                add_tree(tar, Path(root_name), root_name, manifest)
            for file_name in ["pyproject.toml", "README.md"]:
                p = Path(file_name)
                if p.exists():
                    tar.add(p, arcname=p.name)
                    manifest.append({"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        add_tree(tar, artifacts_dir, "artifacts", manifest)
        add_tree(tar, data_dir, "data", manifest)
        add_tree(tar, validation_dir, "outputs/validation", manifest)
        add_tree(tar, logs_dir, "outputs/logs", manifest)
        add_tree(tar, docs_dir, "docs", manifest)
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, fileobj=__import__("io").BytesIO(manifest_bytes))
    summary = {"bundle": str(output), "files": len(manifest), "bytes": output.stat().st_size, "sha256": sha256_file(output)}
    (output.with_suffix(output.suffix + ".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/diffusiongemma-e4b-repro-bundle.tar.gz"))
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.output), indent=2))


if __name__ == "__main__":
    main()
