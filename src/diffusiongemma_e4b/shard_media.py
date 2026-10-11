"""Content-addressed media tensors shared by all answer blocks and shards."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SHARED_MEDIA_KEYS = ("pixel_values", "input_features", "input_features_mask", "image_position_ids")


def tensor_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(str(value.shape).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


class DiskBudget:
    def __init__(self, directory: Path, reserve_bytes: int):
        self.directory = directory
        self.reserve_bytes = reserve_bytes
        self.pending = 0
        self.lock = threading.Lock()

    @contextmanager
    def claim(self, size: int):
        # Allow ZIP/NPY overhead and incompressible arrays, including concurrent writers.
        size = size * 2 + 1024 * 1024
        with self.lock:
            free = shutil.disk_usage(self.directory).free
            if free < self.reserve_bytes + self.pending + size:
                raise OSError("Insufficient disk space for shard conversion; saved checkpoint can be resumed "
                              f"after freeing space (reserve: {self.reserve_bytes / 1024**3:.1f} GiB).")
            self.pending += size
        try:
            yield
        finally:
            with self.lock:
                self.pending -= size


class MediaStore:
    def __init__(self, root: Path, budget: DiskBudget):
        self.directory = root / "media"
        self.directory.mkdir(exist_ok=True)
        self.budget = budget
        self.lock = threading.Lock()
        self.seen: set[str] = set()

    def put(self, key: str, value: np.ndarray) -> str:
        name = f"{key}-{tensor_digest(value)}.npz"
        with self.lock:
            if name in self.seen:
                return name
            path = self.directory / name
            if path.exists():
                with np.load(path, allow_pickle=False) as stored:
                    if tensor_digest(stored["value"]) != tensor_digest(value):
                        raise ValueError(f"Corrupt shared media tensor: {path}")
            else:
                with self.budget.claim(value.nbytes):
                    fd, temporary = tempfile.mkstemp(prefix=".", suffix=".npz", dir=self.directory)
                    os.close(fd)
                    try:
                        np.savez_compressed(temporary, value=value)
                        with open(temporary, "rb") as stream:
                            os.fsync(stream.fileno())
                        os.replace(temporary, path)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
            self.seen.add(name)
        return name


def media_path(root: Path, key: str, reference: str) -> Path:
    if not re.fullmatch(re.escape(key) + r"-[0-9a-f]{64}\.npz", reference):
        raise ValueError(f"Invalid shared media reference: {reference!r}")
    return root / "media" / reference


def load_media_row(root: Path, shard, key: str, row: int) -> np.ndarray | None:
    reference_key = key + "_refs"
    if reference_key in shard:
        path = media_path(root, key, str(shard[reference_key][row]))
        with np.load(path, allow_pickle=False) as stored:
            return stored["value"]
    return shard[key][row] if key in shard else None


def verify_media_references(root: Path, shard, rows: int, seen: set[str]) -> None:
    for key in SHARED_MEDIA_KEYS:
        reference_key = key + "_refs"
        if reference_key not in shard:
            continue
        if key in shard:
            raise ValueError(f"Both embedded and shared media present for {key}")
        references = shard[reference_key]
        if references.shape != (rows,):
            raise ValueError(f"{reference_key} row count mismatch")
        for reference in np.unique(references):
            name = str(reference)
            path = media_path(root, key, name)
            if name in seen:
                continue
            with np.load(path, allow_pickle=False) as stored:
                if tensor_digest(stored["value"]) != name[len(key) + 1:-4]:
                    raise ValueError(f"Corrupt shared media tensor: {path}")
            seen.add(name)
