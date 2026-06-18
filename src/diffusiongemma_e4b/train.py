from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

from .constants import CANVAS_LENGTH


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    tokens_seen: int = 0
    best_val_loss: float | None = None


class CorruptionShardDataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "train", val_fraction: float = 0.01):
        self.files = sorted(data_dir.glob("corruption_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no corruption_*.npz files in {data_dir}")
        self.index: list[tuple[Path, int]] = []
        for path in self.files:
            with np.load(path) as shard:
                n = shard["target_ids"].shape[0]
            val_cut = max(1, int(n * val_fraction))
            rows = range(0, val_cut) if split == "val" else range(val_cut, n)
            self.index.extend((path, i) for i in rows)
        if not self.index:
            raise ValueError(f"empty {split} dataset from {data_dir}")
        self._cache_path: Path | None = None
        self._cache = None

    def __len__(self) -> int:
        return len(self.index)

    def _open(self, path: Path):
        if self._cache_path != path:
            if self._cache is not None:
                self._cache.close()
            self._cache = np.load(path)
            self._cache_path = path
        return self._cache

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, row = self.index[idx]
        shard = self._open(path)
        prefix = shard["prefix_ids"][row].astype(np.int64)
        prefix_len = int(shard["prefix_lens"][row])
        target = shard["target_ids"][row].astype(np.int64)
        corrupted = shard["corrupted_ids"][row].astype(np.int64)
        mask = np.zeros_like(prefix, dtype=np.int64)
        if prefix_len:
            mask[-prefix_len:] = 1
        return {
            "input_ids": torch.from_numpy(prefix),
            "attention_mask": torch.from_numpy(mask),
            "decoder_input_ids": torch.from_numpy(corrupted),
            "labels": torch.from_numpy(target),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def maybe_apply_lora(model, args):
    if args.train_mode == "full":
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PEFT is required for LoRA/QLoRA training. Install with `pip install peft`.") from exc

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)


def load_model(args):
    dtype = getattr(torch, args.dtype)
    quantization_config = None
    if args.train_mode == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Transformers BitsAndBytesConfig is unavailable.") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    kwargs = {
        "dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.model_dir, **kwargs)
    model = maybe_apply_lora(model, args)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def compute_loss(model, batch: dict[str, torch.Tensor], self_conditioning_prob: float) -> torch.Tensor:
    device = next(model.parameters()).device
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    self_conditioning_logits = None
    self_conditioning_mask = None
    if self_conditioning_prob > 0:
        with torch.no_grad():
            first = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
            )
        self_conditioning_logits = first.logits.detach()
        probs = torch.rand(batch["input_ids"].shape[0], device=device) < self_conditioning_prob
        self_conditioning_mask = probs

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
        self_conditioning_logits=self_conditioning_logits,
        self_conditioning_mask=self_conditioning_mask,
    )
    logits = out.logits.float()
    labels = batch["labels"]
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))


def save_checkpoint(model, optimizer, scaler, state: TrainState, output_dir: Path) -> Path:
    ckpt = output_dir / f"checkpoint-{state.step:08d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt, safe_serialization=True)
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), ckpt / "scaler.pt")
    (ckpt / "trainer_state.json").write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt.resolve()), encoding="utf-8")
    return ckpt


def load_state_if_available(optimizer, scaler, output_dir: Path) -> TrainState:
    latest = output_dir / "latest_checkpoint.txt"
    if not latest.exists():
        return TrainState()
    ckpt = Path(latest.read_text(encoding="utf-8").strip())
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        return TrainState()
    state = TrainState(**json.loads(state_path.read_text(encoding="utf-8")))
    opt_path = ckpt / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
    scaler_path = ckpt / "scaler.pt"
    if scaler is not None and scaler_path.exists():
        scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))
    return state


@torch.no_grad()
def validate(model, loader: DataLoader, batches: int, self_conditioning_prob: float) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        losses.append(float(compute_loss(model, batch, self_conditioning_prob=0.0).detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else math.nan


def train(args) -> dict:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    train_ds = CorruptionShardDataset(args.data_dir, split="train", val_fraction=args.val_fraction)
    val_ds = CorruptionShardDataset(args.data_dir, split="val", val_fraction=args.val_fraction)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())
    state = load_state_if_available(optimizer, scaler, output_dir) if args.resume else TrainState()
    log_path = output_dir / "train_log.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=args.max_steps, initial=state.step, desc="train", unit="step")
    while state.step < args.max_steps:
        for batch in train_loader:
            if state.step >= args.max_steps:
                break
            with torch.autocast(device_type="cuda", dtype=getattr(torch, args.dtype), enabled=args.amp and torch.cuda.is_available()):
                loss = compute_loss(model, batch, args.self_conditioning_prob) / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (state.step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            state.step += 1
            state.samples_seen += int(batch["labels"].shape[0])
            state.tokens_seen += int(batch["labels"].numel())
            row = {
                "time": time.time(),
                "step": state.step,
                "loss": float(loss.detach().cpu()) * args.gradient_accumulation_steps,
                "samples_seen": state.samples_seen,
                "tokens_seen": state.tokens_seen,
            }
            if state.step % args.val_interval == 0:
                row["val_loss"] = validate(model, val_loader, args.val_batches, args.self_conditioning_prob)
                if state.best_val_loss is None or row["val_loss"] < state.best_val_loss:
                    state.best_val_loss = row["val_loss"]
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if state.step % args.save_interval == 0:
                save_checkpoint(model, optimizer, scaler, state, output_dir)
            pbar.update(1)
        state.epoch += 1
    pbar.close()
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=True)
    (final_dir / "trainer_state.json").write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    return asdict(state) | {"final_dir": str(final_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
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
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--self-conditioning-prob", type=float, default=0.5)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,gate_up_proj")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
