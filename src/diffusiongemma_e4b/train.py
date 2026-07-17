from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


MULTIMODAL_BATCH_KEYS = (
    "pixel_values",
    "input_features",
    "input_features_mask",
    "image_position_ids",
    "mm_token_type_ids",
)
RECORD_ID_KEYS = ("record_ids", "record_id", "source_record_ids", "source_record_id")
CHECKPOINT_SCHEMA_VERSION = 3


@dataclass
class TrainState:
    # Kept for backward-readable logs: step means consumed micro-batches.
    step: int = 0
    optimizer_steps: int = 0
    epoch: int = 0
    batches_in_epoch: int = 0
    samples_seen: int = 0
    tokens_seen: int = 0
    baseline_val_loss: float | None = None
    latest_val_loss: float | None = None
    best_val_loss: float | None = None
    best_optimizer_steps: int | None = None
    last_val_optimizer_steps: int | None = None
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    data_fingerprint: str | None = None
    training_fingerprint: str | None = None
    gradient_accumulation_steps: int | None = None
    batch_size: int | None = None
    seed: int | None = None


def corruption_data_fingerprint(data_dir: Path) -> str:
    """Hash NPZ member CRCs and sizes without decompressing every tensor."""
    files = sorted(data_dir.glob("corruption_*.npz"))
    if not files:
        raise FileNotFoundError(f"no corruption_*.npz files in {data_dir}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        try:
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.infolist(), key=lambda item: item.filename):
                    digest.update(member.filename.encode("utf-8"))
                    digest.update(str(member.CRC).encode("ascii"))
                    digest.update(str(member.file_size).encode("ascii"))
                    digest.update(str(member.compress_size).encode("ascii"))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"corrupt NPZ shard: {path}") from exc
    for metadata_name in ("corruption_manifest.json", "corruption_progress.json"):
        metadata_path = data_dir / metadata_name
        if metadata_path.exists():
            digest.update(metadata_name.encode("utf-8"))
            digest.update(metadata_path.read_bytes())
    return digest.hexdigest()


def _stable_group_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return str(value)


class CorruptionShardDataset(Dataset):
    """Read shards, split source records, and re-sample uniform-state noise."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        val_fraction: float = 0.01,
        seed: int = 1337,
        *,
        online_corruption: bool = True,
        vocab_size: int | None = None,
        pad_token_id: int = 0,
        noise_min: float = 0.05,
        noise_max: float = 0.95,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"invalid split: {split}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be strictly between 0 and 1")
        if not 0.0 <= noise_min <= noise_max <= 1.0:
            raise ValueError("noise bounds must satisfy 0 <= min <= max <= 1")
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("corruption_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no corruption_*.npz files in {data_dir}")
        self.seed = int(seed)
        self.epoch = 0
        self.online_corruption = bool(online_corruption)
        self.vocab_size = vocab_size
        self.pad_token_id = int(pad_token_id)
        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)

        entries: list[tuple[Path, int, str, str]] = []
        found_record_ids = False
        for path in self.files:
            with np.load(path, allow_pickle=False) as shard:
                target_ids = shard["target_ids"]
                n = int(target_ids.shape[0])
                record_key = next((key for key in RECORD_ID_KEYS if key in shard), None)
                record_values = shard[record_key] if record_key is not None else None
                bucket_values = shard["buckets"] if "buckets" in shard else None
                if record_values is not None and len(record_values) != n:
                    raise ValueError(f"{path}: {record_key} row count mismatch")
                if bucket_values is not None and len(bucket_values) != n:
                    raise ValueError(f"{path}: buckets row count mismatch")
                found_record_ids = found_record_ids or record_key is not None
                for row in range(n):
                    if record_values is None:
                        group = f"legacy:{path.name}:{row}"
                    else:
                        group = f"record:{_stable_group_value(record_values[row])}"
                    bucket = (
                        _stable_group_value(bucket_values[row])
                        if bucket_values is not None
                        else "unknown"
                    )
                    entries.append((path, row, group, bucket))
        if not entries:
            raise ValueError(f"no rows found in {data_dir}")
        if not found_record_ids and split == "train":
            warnings.warn(
                "corruption shards have no record_ids; the legacy per-block split cannot "
                "prevent adjacent response blocks leaking across train/validation",
                stacklevel=2,
            )

        group_buckets: dict[str, str] = {}
        for _, _, group, bucket in entries:
            previous = group_buckets.setdefault(group, bucket)
            if previous != bucket:
                raise ValueError(
                    f"record group {group!r} spans multiple buckets: "
                    f"{previous!r}, {bucket!r}"
                )
        groups = sorted(group_buckets)
        if len(groups) < 2:
            raise ValueError("at least two record groups are required for train/validation")
        group_rng = random.Random(self.seed)
        groups_by_bucket: dict[str, list[str]] = {}
        for group in groups:
            groups_by_bucket.setdefault(group_buckets[group], []).append(group)
        val_groups: set[str] = set()
        for bucket, bucket_groups in sorted(groups_by_bucket.items()):
            if len(bucket_groups) < 2:
                raise ValueError(
                    f"bucket {bucket!r} has only one record group; at least two "
                    "are required for leakage-free stratified train/validation"
                )
            group_rng.shuffle(bucket_groups)
            bucket_val_count = min(
                max(1, int(round(len(bucket_groups) * val_fraction))),
                len(bucket_groups) - 1,
            )
            val_groups.update(bucket_groups[:bucket_val_count])
        selected_entries = [
            (path, row, bucket)
            for path, row, group, bucket in entries
            if (group in val_groups) == (split == "val")
        ]
        self.index = [(path, row) for path, row, _ in selected_entries]
        self.bucket_labels = [bucket for _, _, bucket in selected_entries]
        if not self.index:
            raise ValueError(f"empty {split} dataset from {data_dir}")
        self._cache_path: Path | None = None
        self._cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _open(self, path: Path) -> dict[str, np.ndarray]:
        if self._cache_path != path:
            # NpzFile.__getitem__ decompresses the complete member on every access.
            # Materialize one shard once; the shard-local sampler then reuses these
            # arrays without turning a large training run into O(rows²) decompression.
            with np.load(path, allow_pickle=False) as shard:
                self._cache = {key: shard[key] for key in shard.files}
            self._cache_path = path
        if self._cache is None:
            raise RuntimeError(f"failed to cache corruption shard: {path}")
        return self._cache

    def _row_rng(self, path: Path, row: int) -> np.random.Generator:
        payload = f"{self.seed}:{self.epoch}:{path.name}:{row}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        return np.random.default_rng(seed)

    def _online_noise(
        self,
        target: np.ndarray,
        path: Path,
        row: int,
        fallback_corrupted: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        valid = target >= 0
        if not valid.any():
            raise ValueError(f"{path} row {row} has no target tokens")
        rng = self._row_rng(path, row)
        noise_t = float(rng.uniform(self.noise_min, self.noise_max))
        corruption_mask = (rng.random(target.shape, dtype=np.float32) < noise_t) & valid
        if not corruption_mask.any():
            corruption_mask[int(rng.choice(np.flatnonzero(valid)))] = True
        if self.vocab_size is None:
            vocab_size = max(int(target[valid].max()), int(fallback_corrupted.max()), 0) + 1
        else:
            vocab_size = int(self.vocab_size)
        if vocab_size <= 0:
            raise ValueError(f"invalid vocab size: {vocab_size}")
        corrupted = np.full(target.shape, self.pad_token_id, dtype=np.int64)
        corrupted[valid] = target[valid]
        corrupted[corruption_mask] = rng.integers(
            0, vocab_size, size=int(corruption_mask.sum()), dtype=np.int64
        )
        return corrupted, corruption_mask.astype(np.bool_), noise_t

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, row = self.index[idx]
        shard = self._open(path)
        prefix = shard["prefix_ids"][row].astype(np.int64)
        target = shard["target_ids"][row].astype(np.int64)
        stored_corrupted = shard["corrupted_ids"][row].astype(np.int64)
        attention_mask_key = next(
            (key for key in ("attention_mask", "attention_masks") if key in shard),
            None,
        )
        if attention_mask_key is not None:
            attention_mask = shard[attention_mask_key][row].astype(np.int64)
        else:
            prefix_len = int(shard["prefix_lens"][row])
            attention_mask = np.zeros_like(prefix, dtype=np.int64)
            if prefix_len:
                attention_mask[-prefix_len:] = 1
        if self.online_corruption:
            corrupted, corruption_mask, noise_t = self._online_noise(
                target, path, row, stored_corrupted
            )
        else:
            corrupted = stored_corrupted
            if "corruption_masks" in shard:
                corruption_mask = shard["corruption_masks"][row].astype(np.bool_)
            else:
                corruption_mask = (corrupted != target) & (target >= 0)
            noise_t = (
                float(shard["noise_t"][row])
                if "noise_t" in shard
                else float(corruption_mask[target >= 0].mean())
            )
        return {
            **self._optional_multimodal_tensors(shard, row),
            "input_ids": torch.from_numpy(prefix),
            "attention_mask": torch.from_numpy(attention_mask),
            "decoder_input_ids": torch.from_numpy(corrupted),
            "decoder_attention_mask": torch.from_numpy((target >= 0).astype(np.int64)),
            "labels": torch.from_numpy(target),
            "corruption_mask": torch.from_numpy(corruption_mask),
            "noise_t": torch.tensor(noise_t, dtype=torch.float32),
        }

    def _optional_multimodal_tensors(self, shard, row: int) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for key in MULTIMODAL_BATCH_KEYS:
            if key not in shard:
                continue
            value = shard[key][row]
            dtype = np.float32 if key in {"pixel_values", "input_features"} else np.int64
            tensors[key] = torch.from_numpy(value.astype(dtype))
        return tensors


class DeterministicShardBatchSampler(Sampler[list[int]]):
    """Deterministic batches that preserve shard locality and tensor schemas."""

    def __init__(
        self,
        dataset: CorruptionShardDataset,
        batch_size: int,
        seed: int,
        *,
        shuffle: bool,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.groups: dict[Path, list[int]] = {}
        for dataset_index, (path, _) in enumerate(dataset.index):
            self.groups.setdefault(path, []).append(dataset_index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(math.ceil(len(rows) / self.batch_size) for rows in self.groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        paths = sorted(self.groups, key=str)
        if self.shuffle:
            paths = [paths[i] for i in torch.randperm(len(paths), generator=generator).tolist()]
        for path in paths:
            rows = self.groups[path]
            if self.shuffle:
                rows = [rows[i] for i in torch.randperm(len(rows), generator=generator).tolist()]
            for start in range(0, len(rows), self.batch_size):
                yield rows[start : start + self.batch_size]


class DeterministicValidationBatchSampler(Sampler[list[int]]):
    """Stable proportional held-out rows with every present bucket represented.

    Corruption shards have very different row capacities: an image shard can be
    hundreds of times larger per row than a text shard. This sampler allocates a
    bounded row budget by held-out bucket population, guarantees one row for each
    present bucket, and prefers a small number of shards for bounded decompression.
    Batch size one avoids mixing incompatible multimodal tensor schemas.
    """

    def __init__(
        self,
        dataset: Dataset,
        seed: int,
        max_samples: int | None = None,
    ):
        self.length = len(dataset)
        self.seed = int(seed)
        self.max_samples = min(
            self.length,
            self.length if max_samples is None else int(max_samples),
        )
        if self.max_samples <= 0:
            raise ValueError("validation sampler requires at least one sample")
        labels = list(getattr(dataset, "bucket_labels", []))
        self.bucket_labels = labels if len(labels) == self.length else []
        raw_index = getattr(dataset, "index", [])
        index = list(raw_index) if isinstance(raw_index, (list, tuple)) else []
        self.paths = (
            [Path(path) for path, _ in index]
            if len(index) == self.length
            else []
        )

    def __len__(self) -> int:
        return self.max_samples

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed)
        permutation = torch.randperm(self.length, generator=generator).tolist()
        if not self.bucket_labels:
            selected = permutation[: self.max_samples]
        else:
            bucket_counts: dict[str, int] = {}
            for label in self.bucket_labels:
                bucket_counts[label] = bucket_counts.get(label, 0) + 1
            buckets = sorted(bucket_counts)
            if len(buckets) > self.max_samples:
                raise ValueError(
                    "validation sample budget is smaller than the number of buckets"
                )
            # Reserve one observation per bucket, then distribute the remaining
            # budget proportionally without exceeding any bucket's capacity.
            quotas = {bucket: 1 for bucket in buckets}
            remaining = self.max_samples - len(buckets)
            ideal = {
                bucket: self.max_samples * bucket_counts[bucket] / self.length
                for bucket in buckets
            }
            while remaining:
                eligible = [
                    bucket
                    for bucket in buckets
                    if quotas[bucket] < bucket_counts[bucket]
                ]
                if not eligible:
                    raise RuntimeError(
                        "validation sampler could not allocate its full sample budget"
                    )
                bucket = max(
                    eligible,
                    key=lambda name: (
                        ideal[name] - quotas[name],
                        bucket_counts[name],
                        name,
                    ),
                )
                quotas[bucket] += 1
                remaining -= 1

            rank = {index: position for position, index in enumerate(permutation)}
            selected = []
            for bucket in buckets:
                candidates = [
                    index
                    for index, label in enumerate(self.bucket_labels)
                    if label == bucket
                ]
                if self.paths:
                    by_path: dict[Path, list[int]] = {}
                    for index in candidates:
                        by_path.setdefault(self.paths[index], []).append(index)
                    candidates = [
                        index
                        for path in sorted(
                            by_path,
                            key=lambda item: (-len(by_path[item]), str(item)),
                        )
                        for index in sorted(by_path[path], key=rank.__getitem__)
                    ]
                else:
                    candidates.sort(key=rank.__getitem__)
                selected.extend(candidates[: quotas[bucket]])

        # Keep selected rows path-local so one validation pass decompresses a
        # small bounded number of NPZ members instead of one member per row.
        if self.paths:
            selected.sort(key=lambda index: (str(self.paths[index]), index))
        for index in selected:
            yield [index]


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    expected = set(batch[0])
    for index, item in enumerate(batch[1:], start=1):
        if set(item) != expected:
            raise ValueError(
                f"mixed schemas in one batch: row 0 has {sorted(expected)}, "
                f"row {index} has {sorted(item)}"
            )
    try:
        return {key: torch.stack([item[key] for item in batch]) for key in sorted(expected)}
    except RuntimeError as exc:
        raise ValueError("incompatible tensor shapes within a shard batch") from exc


def _decoder_linear_targets(model: nn.Module, requested: str) -> list[str]:
    requested_names = {
        value.strip() for value in requested.split(",") if value.strip() not in {"", "auto"}
    }
    excluded = {
        "encoder", "vision_tower", "audio_tower", "multi_modal_projector",
        "multimodal_projector", "lm_head",
    }
    targets: list[str] = []
    for name, module in model.named_modules():
        parts = set(name.split("."))
        if not isinstance(module, nn.Linear) or "decoder" not in parts or parts & excluded:
            continue
        if requested_names and name.rsplit(".", 1)[-1] not in requested_names:
            continue
        targets.append(name)
    if not targets:
        suffix = f" matching {sorted(requested_names)}" if requested_names else ""
        raise RuntimeError(
            "no decoder nn.Linear modules found for LoRA" + suffix +
            "; refusing to target encoder or multimodal towers"
        )
    return sorted(set(targets))


def _prepare_kbit_model(model: nn.Module) -> nn.Module:
    try:
        from peft import prepare_model_for_kbit_training
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("QLoRA requires PEFT; install the train extra") from exc
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)


def maybe_apply_lora(model: nn.Module, args) -> nn.Module:
    if args.train_mode == "full":
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("LoRA/QLoRA requires PEFT; install the train extra") from exc
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=_decoder_linear_targets(model, args.lora_target_modules),
        task_type=None,
    )
    model = get_peft_model(model, config)
    for peft_config in getattr(model, "peft_config", {}).values():
        peft_config.base_model_name_or_path = str(args.model_dir)
    return model


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def latest_checkpoint_path(output_dir: Path) -> Path | None:
    latest = output_dir / "latest_checkpoint.txt"
    if not latest.exists():
        return None
    raw = latest.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"empty checkpoint pointer: {latest}")
    path = Path(raw)
    path = path if path.is_absolute() else output_dir / path
    if not path.exists():
        raise RuntimeError(f"checkpoint pointer references missing path: {path}")
    return path


def _load_base_model(
    source: str,
    kwargs: dict[str, Any],
    revision: str | None = None,
) -> tuple[nn.Module, str]:
    import transformers

    config = transformers.AutoConfig.from_pretrained(
        source,
        trust_remote_code=True,
        revision=revision,
    )
    resolved_revision = revision or getattr(config, "_commit_hash", None)
    load_kwargs = dict(kwargs)
    if resolved_revision and not Path(source).exists():
        # Pin the weight fetch to the exact revision that supplied the inspected
        # config. This removes a config/weight race if a Hub branch moves.
        load_kwargs["revision"] = str(resolved_revision)

    def finish(model: nn.Module, loader_kind: str) -> tuple[nn.Module, str]:
        model._diffusiongemma_base_revision = resolved_revision  # type: ignore[attr-defined]
        return model, loader_kind

    architectures = set(getattr(config, "architectures", None) or [])
    if "MultimodalDiffusionGemmaForBlockDiffusion" in architectures:
        from .modeling_multimodal import MultimodalDiffusionGemmaForBlockDiffusion

        model = MultimodalDiffusionGemmaForBlockDiffusion.from_pretrained(source, **load_kwargs)
        return finish(model, "custom_multimodal")
    raise ValueError(
        f"{source} does not declare this project's E4B diffusion architecture; "
        "run the transplant stage and train from artifacts/transplanted"
    )


def _supports_gradient_checkpointing(model: nn.Module) -> bool:
    candidates = [model]
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:  # noqa: BLE001
            pass
    return any(bool(getattr(item, "supports_gradient_checkpointing", False)) for item in candidates)


def load_model(args):
    dtype = getattr(torch, args.dtype)
    quantization_config = None
    if args.train_mode == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA GPU and bitsandbytes")
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("BitsAndBytesConfig is unavailable") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    checkpoint = latest_checkpoint_path(args.output_dir) if args.resume else None
    adapter_checkpoint = checkpoint is not None and (checkpoint / "adapter_config.json").exists()
    if adapter_checkpoint and args.train_mode == "full":
        raise RuntimeError("cannot resume a PEFT adapter with --train-mode full")
    full_checkpoint = (
        checkpoint
        if checkpoint is not None and not adapter_checkpoint and (checkpoint / "config.json").exists()
        else None
    )
    adapter_revision = None
    if adapter_checkpoint:
        adapter_payload = json.loads(
            (checkpoint / "adapter_config.json").read_text(encoding="utf-8")
        )
        adapter_revision = adapter_payload.get("revision")
    model, loader_kind = _load_base_model(
        str(full_checkpoint or args.model_dir),
        kwargs,
        revision=str(adapter_revision) if adapter_revision else None,
    )
    base_revision = getattr(model, "_diffusiongemma_base_revision", None)
    if args.train_mode == "qlora":
        model = _prepare_kbit_model(model)
    if adapter_checkpoint:
        try:
            from peft import PeftModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PEFT is required to resume this adapter") from exc
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=True)
    else:
        model = maybe_apply_lora(model, args)
    model._diffusiongemma_base_revision = base_revision  # type: ignore[attr-defined]
    for peft_config in getattr(model, "peft_config", {}).values():
        peft_config.base_model_name_or_path = str(args.model_dir)
        if base_revision:
            peft_config.revision = str(base_revision)

    if args.gradient_checkpointing:
        if _supports_gradient_checkpointing(model):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model, "config"):
                model.config.use_cache = False
        else:
            warnings.warn(
                "gradient checkpointing was requested, but this DiffusionGemma class "
                "declares it unsupported; continuing with it disabled",
                stacklevel=2,
            )
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("model has no trainable parameters")
    model._diffusiongemma_loader_kind = loader_kind  # type: ignore[attr-defined]
    return model


def _input_device(model: nn.Module) -> torch.device:
    try:
        device = model.get_input_embeddings().weight.device
        if device.type != "meta":
            return device
    except Exception:  # noqa: BLE001
        pass
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("could not determine model input device")


def _forward_kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    canvas_mask = batch.get("decoder_attention_mask")
    if canvas_mask is None:
        canvas_mask = batch["labels"].ne(-100).long()
    prefix_length = batch["attention_mask"].shape[-1]
    canvas_length = batch["decoder_input_ids"].shape[-1]
    if canvas_mask.shape[-1] == canvas_length:
        decoder_mask = torch.cat([batch["attention_mask"], canvas_mask], dim=-1)
    elif canvas_mask.shape[-1] == prefix_length + canvas_length:
        decoder_mask = canvas_mask
    else:
        raise ValueError(
            "decoder_attention_mask must cover the canvas or encoder+canvas; got "
            f"{canvas_mask.shape[-1]} for {prefix_length}+{canvas_length} tokens"
        )
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "decoder_input_ids": batch["decoder_input_ids"],
        "decoder_attention_mask": decoder_mask,
    }
    kwargs.update({key: batch[key] for key in MULTIMODAL_BATCH_KEYS if key in batch})
    return kwargs


def compute_loss(
    model,
    batch: dict[str, torch.Tensor],
    self_conditioning_prob: float,
    clean_token_loss_weight: float = 0.0,
) -> torch.Tensor:
    if not 0.0 <= clean_token_loss_weight <= 1.0:
        raise ValueError("clean_token_loss_weight must be in [0, 1]")
    device = _input_device(model)
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    kwargs = _forward_kwargs(batch)
    self_conditioning_logits = None
    self_conditioning_mask = None
    if self_conditioning_prob > 0:
        self_conditioning_mask = (
            torch.rand(batch["decoder_input_ids"].shape[0], device=device)
            < self_conditioning_prob
        )
        if self_conditioning_mask.any():
            with torch.no_grad():
                first = model(**kwargs)
            self_conditioning_logits = first.logits.detach()
            del first
    output = model(
        **kwargs,
        self_conditioning_logits=self_conditioning_logits,
        self_conditioning_mask=self_conditioning_mask,
    )
    logits = output.logits.float()
    labels = batch["labels"]
    valid = labels.ne(-100)
    corruption_mask = batch.get("corruption_mask")
    corruption_mask = valid if corruption_mask is None else corruption_mask.bool() & valid
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    weights = corruption_mask.to(per_token.dtype)
    if clean_token_loss_weight:
        weights = weights + (valid & ~corruption_mask).to(per_token.dtype) * clean_token_loss_weight
    denominator = weights.sum()
    if denominator.item() <= 0:
        raise ValueError("batch has no weighted target tokens")
    return (per_token * weights).sum() / denominator


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("GPU count changed; exact CUDA RNG resume is impossible")
        torch.cuda.set_rng_state_all(state["cuda"])


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _remove_old_checkpoints(output_dir: Path, limit: int) -> None:
    if limit <= 0:
        return
    checkpoints = sorted(
        path for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and ".tmp-" not in path.name
    )
    for old in checkpoints[:-limit]:
        shutil.rmtree(old)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    state: TrainState,
    output_dir: Path,
    *,
    gradient_accumulation_steps: int,
    save_total_limit: int,
) -> Path:
    if state.step % gradient_accumulation_steps:
        raise RuntimeError("refusing to checkpoint mid gradient accumulation")
    checkpoint = output_dir / f"checkpoint-{state.optimizer_steps:08d}"
    temporary = output_dir / f".{checkpoint.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    torch.save(scaler.state_dict(), temporary / "scaler.pt")
    torch.save(_capture_rng_state(), temporary / "rng_state.pt")
    (temporary / "trainer_state.json").write_text(
        json.dumps(asdict(state), indent=2), encoding="utf-8"
    )
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    os.replace(temporary, checkpoint)
    _atomic_write_text(output_dir / "latest_checkpoint.txt", checkpoint.name)
    _remove_old_checkpoints(output_dir, save_total_limit)
    return checkpoint


def _state_from_json(payload: dict[str, Any]) -> TrainState:
    allowed = {field.name for field in fields(TrainState)}
    return TrainState(**{key: value for key, value in payload.items() if key in allowed})


def load_state_if_available(
    optimizer,
    scheduler,
    scaler,
    output_dir: Path,
    *,
    expected_data_fingerprint: str,
    expected_training_fingerprint: str,
    gradient_accumulation_steps: int,
) -> TrainState:
    checkpoint = latest_checkpoint_path(output_dir)
    if checkpoint is None:
        return TrainState()
    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        raise RuntimeError(f"checkpoint lacks trainer_state.json: {checkpoint}")
    state = _state_from_json(json.loads(state_path.read_text(encoding="utf-8")))
    if state.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema is not exact-resume compatible")
    if state.data_fingerprint != expected_data_fingerprint:
        raise RuntimeError("corruption dataset changed; refusing unsafe resume")
    if state.training_fingerprint != expected_training_fingerprint:
        raise RuntimeError("training hyperparameters changed; refusing unsafe resume")
    if state.gradient_accumulation_steps != gradient_accumulation_steps:
        raise RuntimeError("gradient accumulation changed since checkpoint")
    if state.step % gradient_accumulation_steps:
        raise RuntimeError("checkpoint was written mid accumulation")
    required = ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pt")
    missing = [name for name in required if not (checkpoint / name).exists()]
    if missing:
        raise RuntimeError(f"checkpoint lacks exact-resume state: {missing}")
    optimizer.load_state_dict(_torch_load(checkpoint / "optimizer.pt"))
    scheduler.load_state_dict(_torch_load(checkpoint / "scheduler.pt"))
    scaler.load_state_dict(_torch_load(checkpoint / "scaler.pt"))
    _restore_rng_state(_torch_load(checkpoint / "rng_state.pt"))
    return state


def _resolve_warmup_steps(args) -> int:
    if args.warmup_steps is not None:
        if args.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        return min(args.warmup_steps, args.max_optimizer_steps)
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    return int(round(args.max_optimizer_steps * args.warmup_ratio))


def _cosine_scheduler(optimizer, total_steps: int, warmup_steps: int):
    if total_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _training_fingerprint(args, warmup_steps: int) -> str:
    contract = {
        "model_dir": str(args.model_dir),
        "tokenizer_source": str(args.tokenizer_source or args.model_dir),
        "train_mode": args.train_mode,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "amp": args.amp,
        "gradient_checkpointing": args.gradient_checkpointing,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "max_optimizer_steps": args.max_optimizer_steps,
        "warmup_steps": warmup_steps,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "val_interval": args.val_interval,
        "val_batches": args.val_batches,
        "min_relative_val_improvement": args.min_relative_val_improvement,
        "online_corruption": not args.offline_corruption,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
        "clean_token_loss_weight": args.clean_token_loss_weight,
        "self_conditioning_prob": args.self_conditioning_prob,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_vocab_size(model) -> int:
    candidate = model
    if hasattr(model, "get_base_model"):
        try:
            candidate = model.get_base_model()
        except Exception:  # noqa: BLE001
            pass
    config = candidate.config
    text_config = getattr(config, "text_config", config)
    vocab_size = int(text_config.vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"invalid model vocabulary size: {vocab_size}")
    return vocab_size


def _load_tokenizer_and_processor(args):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    source = str(args.tokenizer_source or args.model_dir)
    tokenizer_config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    tokenizer_revision = getattr(tokenizer_config, "_commit_hash", None)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=True,
        revision=tokenizer_revision,
    )
    tokenizer._diffusiongemma_revision = tokenizer_revision  # type: ignore[attr-defined]
    processor = None
    processor_error = None
    try:
        processor = AutoProcessor.from_pretrained(
            source,
            trust_remote_code=True,
            revision=tokenizer_revision,
        )
    except Exception as exc:  # noqa: BLE001
        processor_error = str(exc)
        warnings.warn(f"AutoProcessor could not be loaded from {source}: {exc}", stacklevel=2)
    return tokenizer, processor, processor_error


@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    batches: int,
    clean_token_loss_weight: float = 0.0,
) -> float:
    model.eval()
    losses = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        loss = compute_loss(
            model,
            batch,
            self_conditioning_prob=0.0,
            clean_token_loss_weight=clean_token_loss_weight,
        )
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else math.nan


def _replace_directory_atomically(temporary: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _save_best_model(model, state: TrainState, output_dir: Path) -> Path:
    best_dir = output_dir / "best"
    temporary = output_dir / f".best.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    (temporary / "best_state.json").write_text(
        json.dumps(asdict(state), indent=2), encoding="utf-8"
    )
    _replace_directory_atomically(temporary, best_dir)
    return best_dir


def _reconcile_best_snapshot(state: TrainState, output_dir: Path) -> None:
    """Reconcile a best snapshot that may be newer than the last optimizer checkpoint."""

    state_path = output_dir / "best" / "best_state.json"
    if not state_path.is_file():
        return
    best_state = _state_from_json(json.loads(state_path.read_text(encoding="utf-8")))
    if best_state.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("best snapshot schema is not compatible with this trainer")
    if (
        best_state.data_fingerprint != state.data_fingerprint
        or best_state.training_fingerprint != state.training_fingerprint
    ):
        raise RuntimeError("best snapshot belongs to different data or hyperparameters")
    if best_state.best_val_loss is None or not math.isfinite(best_state.best_val_loss):
        raise RuntimeError("best snapshot has no finite validation loss")
    if state.best_val_loss is None or best_state.best_val_loss <= state.best_val_loss:
        state.baseline_val_loss = (
            state.baseline_val_loss
            if state.baseline_val_loss is not None
            else best_state.baseline_val_loss
        )
        state.best_val_loss = best_state.best_val_loss
        state.best_optimizer_steps = best_state.best_optimizer_steps


def _relative_validation_improvement(state: TrainState) -> float:
    if state.baseline_val_loss is None or state.best_val_loss is None:
        raise RuntimeError("baseline/best validation loss is missing")
    if not math.isfinite(state.baseline_val_loss) or state.baseline_val_loss <= 0:
        raise RuntimeError(f"invalid baseline validation loss: {state.baseline_val_loss}")
    return (state.baseline_val_loss - state.best_val_loss) / state.baseline_val_loss


def _save_final(model, tokenizer, processor, processor_error, state: TrainState, args) -> Path:
    final_dir = args.output_dir / "final"
    best_dir = args.output_dir / "best"
    if not best_dir.is_dir():
        raise RuntimeError("best validation snapshot is missing; refusing to export the last step blindly")
    temporary = args.output_dir / f".final.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(best_dir, temporary)
    tokenizer.save_pretrained(temporary)
    if processor is not None:
        processor.save_pretrained(temporary)
    elif processor_error:
        (temporary / "processor_save_warning.txt").write_text(
            processor_error, encoding="utf-8"
        )
    (temporary / "trainer_state.json").write_text(
        json.dumps(asdict(state), indent=2), encoding="utf-8"
    )
    adapter_only = (temporary / "adapter_config.json").exists()
    relative_improvement = _relative_validation_improvement(state)
    metadata = {
        "format": "peft_adapter" if adapter_only else "full_model",
        "base_model_name_or_path": str(args.model_dir),
        "tokenizer_source": str(args.tokenizer_source or args.model_dir),
        "train_mode": args.train_mode,
        "dtype": args.dtype,
        "loader_kind": getattr(model, "_diffusiongemma_loader_kind", None),
        "base_model_revision": getattr(model, "_diffusiongemma_base_revision", None),
        "tokenizer_revision": getattr(tokenizer, "_diffusiongemma_revision", None),
        "data_fingerprint": state.data_fingerprint,
        "training_fingerprint": state.training_fingerprint,
        "selection": "lowest deterministic held-out denoising loss",
        "baseline_val_loss": state.baseline_val_loss,
        "latest_val_loss": state.latest_val_loss,
        "best_val_loss": state.best_val_loss,
        "best_optimizer_steps": state.best_optimizer_steps,
        "relative_val_improvement": relative_improvement,
        "minimum_relative_val_improvement": args.min_relative_val_improvement,
        "adapter_reload": (
            "Load base_model_name_or_path with the same quantization config, then "
            "PeftModel.from_pretrained(base, this_directory)."
            if adapter_only else None
        ),
    }
    (temporary / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _replace_directory_atomically(temporary, final_dir)
    return final_dir


def train(args) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.save_interval < 0 or args.val_interval < 0 or args.save_total_limit < 0:
        raise ValueError("save/validation intervals and save_total_limit cannot be negative")
    if args.val_batches <= 0:
        raise ValueError("val_batches must be positive so best-model selection is meaningful")
    if not 0.0 <= args.min_relative_val_improvement < 1.0:
        raise ValueError("min_relative_val_improvement must be in [0, 1)")
    if not args.resume and (args.output_dir / "latest_checkpoint.txt").exists():
        raise RuntimeError(
            f"{args.output_dir} already contains checkpoints; use --resume or a new output directory"
        )
    if not 0.0 <= args.self_conditioning_prob <= 1.0:
        raise ValueError("self_conditioning_prob must be in [0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_fingerprint = corruption_data_fingerprint(args.data_dir)
    warmup_steps = _resolve_warmup_steps(args)
    training_fingerprint = _training_fingerprint(args, warmup_steps)
    tokenizer, processor, processor_error = _load_tokenizer_and_processor(args)
    model = load_model(args)
    vocab_size = _model_vocab_size(model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    dataset_kwargs = {
        "data_dir": args.data_dir,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "online_corruption": not args.offline_corruption,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
    }
    train_dataset = CorruptionShardDataset(split="train", **dataset_kwargs)
    val_dataset = CorruptionShardDataset(split="val", **dataset_kwargs)
    val_dataset.set_epoch(0)
    train_sampler = DeterministicShardBatchSampler(
        train_dataset, args.batch_size, args.seed, shuffle=True
    )
    val_sampler = DeterministicValidationBatchSampler(
        val_dataset,
        args.seed + 1_000_000,
        max_samples=args.val_batches,
    )
    train_worker_generator = torch.Generator()
    val_worker_generator = torch.Generator().manual_seed(args.seed + 1_000_000)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        generator=train_worker_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        generator=val_worker_generator,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = _cosine_scheduler(optimizer, args.max_optimizer_steps, warmup_steps)
    use_scaler = args.amp and torch.cuda.is_available() and args.dtype == "float16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    state = (
        load_state_if_available(
            optimizer,
            scheduler,
            scaler,
            args.output_dir,
            expected_data_fingerprint=data_fingerprint,
            expected_training_fingerprint=training_fingerprint,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        if args.resume else TrainState()
    )
    if state.optimizer_steps > args.max_optimizer_steps:
        raise RuntimeError("checkpoint exceeds max_optimizer_steps")
    if state.batch_size not in {None, args.batch_size} or state.seed not in {None, args.seed}:
        raise RuntimeError("checkpoint batch size or seed differs")
    state.data_fingerprint = data_fingerprint
    state.training_fingerprint = training_fingerprint
    state.gradient_accumulation_steps = args.gradient_accumulation_steps
    state.batch_size = args.batch_size
    state.seed = args.seed
    state.schema_version = CHECKPOINT_SCHEMA_VERSION
    _reconcile_best_snapshot(state, args.output_dir)

    best_dir = args.output_dir / "best"
    if state.best_val_loss is None:
        if state.optimizer_steps != 0:
            raise RuntimeError("resumed checkpoint has no best validation loss")
        baseline_loss = validate(
            model,
            val_loader,
            args.val_batches,
            clean_token_loss_weight=args.clean_token_loss_weight,
        )
        if not math.isfinite(baseline_loss):
            raise FloatingPointError(f"non-finite baseline validation loss: {baseline_loss}")
        state.baseline_val_loss = baseline_loss
        state.latest_val_loss = baseline_loss
        state.best_val_loss = baseline_loss
        state.best_optimizer_steps = 0
        state.last_val_optimizer_steps = 0
        _save_best_model(model, state, args.output_dir)
    elif not best_dir.is_dir():
        raise RuntimeError(
            "checkpoint records a best validation loss but the best snapshot is missing; "
            "refusing to replace it with a later model"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_path = args.output_dir / "train_log.jsonl"
    accumulated_loss = 0.0
    accumulated_batches = 0
    progress = tqdm(
        total=args.max_optimizer_steps,
        initial=state.optimizer_steps,
        desc="train",
        unit="update",
    )
    while state.optimizer_steps < args.max_optimizer_steps:
        train_dataset.set_epoch(state.epoch)
        train_sampler.set_epoch(state.epoch)
        train_worker_generator.manual_seed(args.seed + state.epoch)
        if state.batches_in_epoch > len(train_sampler):
            raise RuntimeError("checkpoint batch cursor exceeds epoch length")
        completed_epoch = True
        for batch_index, batch in enumerate(train_loader):
            if batch_index < state.batches_in_epoch:
                continue
            device = _input_device(model)
            amp_enabled = (
                args.amp and device.type == "cuda" and args.dtype in {"float16", "bfloat16"}
            )
            with torch.autocast(
                device_type=device.type,
                dtype=getattr(torch, args.dtype),
                enabled=amp_enabled,
            ):
                raw_loss = compute_loss(
                    model,
                    batch,
                    args.self_conditioning_prob,
                    clean_token_loss_weight=args.clean_token_loss_weight,
                )
                loss = raw_loss / args.gradient_accumulation_steps
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(
                    f"non-finite loss at micro-step {state.step}: "
                    f"{float(raw_loss.detach().cpu())}"
                )
            scaler.scale(loss).backward()
            state.step += 1
            state.batches_in_epoch = batch_index + 1
            state.samples_seen += int(batch["labels"].shape[0])
            state.tokens_seen += int(batch["labels"].ne(-100).sum().item())
            accumulated_loss += float(raw_loss.detach().cpu())
            accumulated_batches += 1
            if state.step % args.gradient_accumulation_steps:
                continue

            scaler.unscale_(optimizer)
            max_norm = args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state.optimizer_steps += 1
            row = {
                "time": time.time(),
                "step": state.step,
                "optimizer_steps": state.optimizer_steps,
                "epoch": state.epoch,
                "batches_in_epoch": state.batches_in_epoch,
                "loss": accumulated_loss / max(1, accumulated_batches),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "grad_norm": float(grad_norm.detach().cpu()),
                "samples_seen": state.samples_seen,
                "tokens_seen": state.tokens_seen,
            }
            accumulated_loss = 0.0
            accumulated_batches = 0
            if args.val_interval and state.optimizer_steps % args.val_interval == 0:
                row["val_loss"] = validate(
                    model,
                    val_loader,
                    args.val_batches,
                    clean_token_loss_weight=args.clean_token_loss_weight,
                )
                if not math.isfinite(row["val_loss"]):
                    raise FloatingPointError(
                        f"non-finite validation loss at optimizer step {state.optimizer_steps}: "
                        f"{row['val_loss']}"
                    )
                state.latest_val_loss = row["val_loss"]
                state.last_val_optimizer_steps = state.optimizer_steps
                if state.best_val_loss is None or row["val_loss"] < state.best_val_loss:
                    state.best_val_loss = row["val_loss"]
                    state.best_optimizer_steps = state.optimizer_steps
                    _save_best_model(model, state, args.output_dir)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(row) + "\n")
            if args.save_interval and state.optimizer_steps % args.save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    state,
                    args.output_dir,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    save_total_limit=args.save_total_limit,
                )
            progress.update(1)
            if state.optimizer_steps >= args.max_optimizer_steps:
                completed_epoch = False
                break
        if completed_epoch:
            state.epoch += 1
            state.batches_in_epoch = 0
    progress.close()

    if state.last_val_optimizer_steps != state.optimizer_steps:
        final_val_loss = validate(
            model,
            val_loader,
            args.val_batches,
            clean_token_loss_weight=args.clean_token_loss_weight,
        )
        if not math.isfinite(final_val_loss):
            raise FloatingPointError(
                f"non-finite final validation loss at optimizer step {state.optimizer_steps}: "
                f"{final_val_loss}"
            )
        state.latest_val_loss = final_val_loss
        state.last_val_optimizer_steps = state.optimizer_steps
        if state.best_val_loss is None or final_val_loss < state.best_val_loss:
            state.best_val_loss = final_val_loss
            state.best_optimizer_steps = state.optimizer_steps
            _save_best_model(model, state, args.output_dir)

    latest = latest_checkpoint_path(args.output_dir)
    expected_name = f"checkpoint-{state.optimizer_steps:08d}"
    if latest is None or latest.name != expected_name:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            state,
            args.output_dir,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=args.save_total_limit,
        )
    relative_improvement = _relative_validation_improvement(state)
    if relative_improvement + 1e-12 < args.min_relative_val_improvement:
        raise RuntimeError(
            "distillation release gate failed: best held-out denoising loss improved "
            f"{relative_improvement:.6%} from baseline, below the required "
            f"{args.min_relative_val_improvement:.6%}. The best snapshot is preserved "
            "for diagnosis, but no final artifact will be exported."
        )
    final_dir = _save_final(
        model, tokenizer, processor, processor_error, state, args
    )
    return asdict(state) | {"final_dir": str(final_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Local E4B diffusion model produced by the transplant stage",
    )
    parser.add_argument("--tokenizer-source", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/corruption"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/conversion_training"))
    parser.add_argument("--train-mode", choices=["lora", "qlora", "full"], default="lora")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-optimizer-steps",
        "--max-steps",
        dest="max_optimizer_steps",
        type=int,
        default=200_000,
        help="Optimizer updates; --max-steps is a deprecated alias",
    )
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument(
        "--save-total-limit",
        "--checkpoint-retention",
        dest="save_total_limit",
        type=int,
        default=3,
    )
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--min-relative-val-improvement", type=float, default=0.001)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--self-conditioning-prob", type=float, default=0.5)
    parser.add_argument("--clean-token-loss-weight", type=float, default=0.0)
    parser.add_argument("--noise-min", type=float, default=0.05)
    parser.add_argument("--noise-max", type=float, default=0.95)
    parser.add_argument("--offline-corruption", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="auto",
        help="auto discovers decoder nn.Linear modules; or pass decoder leaf names",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
