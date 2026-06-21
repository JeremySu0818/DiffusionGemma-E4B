from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch

from .config import build_diffusion_e4b_config


def run(cmd: list[str], timeout: int = 30) -> dict:
    try:
        p = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc)}


def package_status() -> dict:
    import importlib.util

    names = ["torch", "transformers", "accelerate", "peft", "bitsandbytes", "safetensors", "requests", "numpy", "tqdm"]
    return {name: importlib.util.find_spec(name) is not None for name in names}


def gpu_status() -> dict:
    status = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        status.update(
            {
                "name": props.name,
                "total_vram_gb": round(props.total_memory / 1024**3, 3),
                "torch_cuda": torch.version.cuda,
            }
        )
    return status


def estimate_training_memory(base_model: str) -> dict:
    cfg = build_diffusion_e4b_config(base_model)
    text = cfg.text_config
    try:
        from accelerate import init_empty_weights

        from .modeling_multimodal import MultimodalDiffusionGemmaForBlockDiffusion

        with init_empty_weights():
            model = MultimodalDiffusionGemmaForBlockDiffusion(cfg)
        min_params = sum(p.numel() for p in model.parameters())
        param_source = "meta_initialized_multimodal_diffusion_student"
    except Exception:
        min_params = 7.5e9 if text.hidden_size == 2560 and text.num_hidden_layers == 42 else text.hidden_size * text.hidden_size * text.num_hidden_layers * 12
        param_source = "fallback_estimate"
    bf16_weights_gb = min_params * 2 / 1024**3
    qlora_floor_gb = min_params * 0.58 / 1024**3
    activation_floor_gb = text.num_hidden_layers * 256 * text.hidden_size * 2 * 8 / 1024**3
    return {
        "hidden_size": text.hidden_size,
        "layers": text.num_hidden_layers,
        "vocab_size": text.vocab_size,
        "parameter_source": param_source,
        "min_params_estimate": int(min_params),
        "bf16_weight_floor_gb": round(bf16_weights_gb, 2),
        "qlora_weight_floor_gb": round(qlora_floor_gb, 2),
        "activation_floor_batch1_seq256_gb": round(activation_floor_gb, 2),
        "recommended_large_run_vram_gb": 80,
    }


def cuda_allocation_probe(target_gb: float) -> dict:
    if not torch.cuda.is_available():
        return {"attempted": False, "reason": "cuda unavailable"}
    torch.cuda.empty_cache()
    elems = int(target_gb * 1024**3 / 2)
    started = time.time()
    try:
        x = torch.empty((elems,), dtype=torch.bfloat16, device="cuda")
        x.fill_(0)
        del x
        torch.cuda.empty_cache()
        return {"attempted": True, "target_gb": target_gb, "ok": True, "seconds": round(time.time() - started, 3)}
    except Exception as exc:  # noqa: BLE001
        torch.cuda.empty_cache()
        return {"attempted": True, "target_gb": target_gb, "ok": False, "error": repr(exc), "seconds": round(time.time() - started, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-4-E4B-it")
    parser.add_argument("--output", type=Path, default=Path("outputs/local_probe/local_feasibility_report.json"))
    parser.add_argument("--allocation-gb", type=float, default=7.0)
    args = parser.parse_args()

    report = {
        "packages": package_status(),
        "gpu": gpu_status(),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout=10),
        "ollama_list": run(["ollama", "list"], timeout=20) if shutil.which("ollama") else {"available": False},
        "lmstudio_list": run(["lms", "ls"], timeout=30) if shutil.which("lms") else {"available": False},
        "training_memory_estimate": estimate_training_memory(args.base_model),
        "cuda_allocation_probe": cuda_allocation_probe(args.allocation_gb),
    }
    vram = report["gpu"].get("total_vram_gb", 0)
    recommended = report["training_memory_estimate"]["recommended_large_run_vram_gb"]
    report["large_run_capacity_ok"] = bool(vram >= recommended)
    report["gpu_capacity_status"] = "large_run_ready" if report["large_run_capacity_ok"] else "capacity_limited_for_large_run"
    report["notes"] = [
        "This probe reports local environment capacity only.",
        "It does not select or require any cloud provider.",
        "Lower-memory experiments can still run by reducing batch size, using QLoRA, or shrinking step/sample counts.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
