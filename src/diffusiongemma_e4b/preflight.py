from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .pipeline_presets import PRESETS


GIB = 1024**3
REQUIRED_IMPORTS = ("numpy", "requests", "torch", "transformers", "datasets", "huggingface_hub")


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    message: str


def _integer(env: Mapping[str, str], key: str, default: int, findings: list[Finding], minimum: int = 0) -> int:
    raw = env.get(key, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        findings.append(Finding("error", "invalid_integer", f"{key} must be an integer; got {raw!r}."))
        return default
    if value < minimum:
        findings.append(Finding("error", "invalid_range", f"{key} must be >= {minimum}; got {value}."))
    return value


def _number(env: Mapping[str, str], key: str, default: float, findings: list[Finding], minimum: float = 0.0) -> float:
    raw = env.get(key, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        findings.append(Finding("error", "invalid_number", f"{key} must be numeric; got {raw!r}."))
        return default
    if value < minimum:
        findings.append(Finding("error", "invalid_range", f"{key} must be >= {minimum}; got {value}."))
    return value


def validate_numeric_settings(env: Mapping[str, str]) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    target_tokens = _integer(env, "DG_TARGET_ESTIMATED_TOKENS", 0, findings, 1)
    target_blocks = _integer(env, "DG_TARGET_BLOCKS", 0, findings, 1)
    canvas = _integer(env, "DG_CANVAS_LENGTH", 256, findings, 1)
    prefix = _integer(env, "DG_PREFIX_LENGTH", 512, findings, 1)
    batch = _integer(env, "DG_BATCH_SIZE", 1, findings, 1)
    accum = _integer(env, "DG_GRAD_ACCUM", 32, findings, 1)
    optimizer_steps = _integer(env, "DG_MAX_OPTIMIZER_STEPS", 0, findings, 1)
    warmup = _integer(env, "DG_WARMUP_STEPS", 0, findings, 0)
    save_interval = _integer(env, "DG_SAVE_INTERVAL", 1000, findings, 1)
    val_interval = _integer(env, "DG_VAL_INTERVAL", 500, findings, 1)
    retention = _integer(env, "DG_CHECKPOINT_RETENTION", 3, findings, 1)
    teacher_start_timeout = _integer(env, "DG_TEACHER_START_TIMEOUT_S", 1800, findings, 1)
    teacher_concurrency = _integer(env, "DG_TEACHER_CONCURRENCY", 8, findings, 1)
    teacher_tensor_parallel = _integer(
        env, "DG_TEACHER_TENSOR_PARALLEL_SIZE", 1, findings, 1
    )
    teacher_max_model_len = _integer(
        env, "DG_TEACHER_MAX_MODEL_LEN", 16_384, findings, 1
    )
    max_prompt_chars = _integer(env, "DG_MAX_PROMPT_CHARS", 11_000, findings, 1)
    max_tokens_per_sample = _integer(
        env, "DG_MAX_TOKENS_PER_SAMPLE", 4_096, findings, 1
    )
    teacher_gpu_memory_utilization = _number(
        env, "DG_TEACHER_GPU_MEMORY_UTILIZATION", 0.80, findings, 0.0
    )
    lr = _number(env, "DG_LEARNING_RATE", _number(env, "DG_LR", 1e-4, findings), findings, 0.0)
    weight_decay = _number(env, "DG_WEIGHT_DECAY", 0.1, findings, 0.0)
    min_relative_val_improvement = _number(
        env, "DG_MIN_RELATIVE_VAL_IMPROVEMENT", 0.001, findings, 0.0
    )

    block_token_capacity = target_blocks * canvas
    if canvas != 256:
        findings.append(
            Finding(
                "error",
                "e4b_canvas_length_mismatch",
                "The E4B diffusion architecture requires DG_CANVAS_LENGTH=256.",
            )
        )
    if prefix < 512:
        findings.append(
            Finding(
                "error",
                "e4b_prefix_too_short",
                "DG_PREFIX_LENGTH must be at least 512 so image soft tokens, chat markup, "
                "and useful text conditioning fit in the E4B student prefix.",
            )
        )
    # Corruption must write exactly target_blocks.  Requiring at least one
    # teacher token per available canvas position is a conservative guarantee:
    # response-boundary padding can only increase the number of available blocks.
    if target_tokens < block_token_capacity:
        findings.append(
            Finding(
                "error",
                "insufficient_teacher_tokens",
                f"DG_TARGET_ESTIMATED_TOKENS is {target_tokens:,}, below the exact "
                f"DG_TARGET_BLOCKS x DG_CANVAS_LENGTH capacity of {block_token_capacity:,}. "
                "Increase the teacher-token target or reduce DG_TARGET_BLOCKS.",
            )
        )
    elif target_tokens > int(block_token_capacity * 1.20):
        findings.append(
            Finding(
                "warning",
                "excess_teacher_tokens",
                f"The teacher target is more than 20% above the {block_token_capacity:,}-token "
                "training-canvas capacity; some generated targets will not be consumed.",
            )
        )
    if warmup >= optimizer_steps and optimizer_steps > 0:
        findings.append(Finding("error", "warmup_too_long", "DG_WARMUP_STEPS must be smaller than DG_MAX_OPTIMIZER_STEPS."))
    if save_interval > optimizer_steps and optimizer_steps > 0:
        findings.append(Finding("warning", "no_periodic_checkpoint", "Save interval exceeds the whole run; only the final artifact would be saved."))
    if val_interval > optimizer_steps and optimizer_steps > 0:
        findings.append(Finding("warning", "no_periodic_validation", "Validation interval exceeds the whole run."))
    if lr <= 0:
        findings.append(Finding("error", "invalid_learning_rate", "Learning rate must be greater than zero."))
    if weight_decay > 1:
        findings.append(Finding("error", "invalid_weight_decay", "DG_WEIGHT_DECAY must be in [0, 1]."))
    if not 0.0 < teacher_gpu_memory_utilization < 1.0:
        findings.append(
            Finding(
                "error",
                "invalid_teacher_gpu_memory_utilization",
                "DG_TEACHER_GPU_MEMORY_UTILIZATION must be strictly between 0 and 1.",
            )
        )
    # A Unicode character can be a token, so chars + requested completion is a
    # safe upper bound that prevents vLLM context-length rejection mid-run.
    teacher_context_reserve = 512
    if (
        teacher_max_model_len
        < max_prompt_chars + max_tokens_per_sample + teacher_context_reserve
    ):
        findings.append(
            Finding(
                "error",
                "teacher_context_too_short",
                "DG_TEACHER_MAX_MODEL_LEN must cover DG_MAX_PROMPT_CHARS + "
                "DG_MAX_TOKENS_PER_SAMPLE plus 512 reserved image/chat-template tokens.",
            )
        )
    if min_relative_val_improvement >= 1.0:
        findings.append(
            Finding(
                "error",
                "invalid_min_relative_val_improvement",
                "DG_MIN_RELATIVE_VAL_IMPROVEMENT must be in [0, 1).",
            )
        )
    if env.get("DG_GRADIENT_CHECKPOINTING") == "1":
        findings.append(
            Finding(
                "error",
                "e4b_cache_gradient_checkpointing_unsupported",
                "Gradient checkpointing would drop the encoder KV cache required "
                "for prompt-conditioned diffusion training. Keep "
                "DG_GRADIENT_CHECKPOINTING=0.",
            )
        )
    if env.get("DG_ALLOW_PARTIAL_TRANSPLANT") == "1":
        findings.append(
            Finding(
                "error",
                "partial_e4b_transplant_forbidden",
                "The production pipeline requires complete E4B weight coverage; "
                "DG_ALLOW_PARTIAL_TRANSPLANT=1 is not supported.",
            )
        )
    return {
        "target_estimated_tokens": target_tokens,
        "target_blocks": target_blocks,
        "block_token_capacity": block_token_capacity,
        "canvas_length": canvas,
        "prefix_length": prefix,
        "batch_size": batch,
        "gradient_accumulation_steps": accum,
        "effective_batch_blocks": batch * accum,
        "planned_block_passes": round(
            optimizer_steps * batch * accum / max(target_blocks, 1), 4
        ),
        "max_optimizer_steps": optimizer_steps,
        "warmup_steps": warmup,
        "save_interval": save_interval,
        "val_interval": val_interval,
        "checkpoint_retention": retention,
        "teacher_start_timeout_s": teacher_start_timeout,
        "teacher_concurrency": teacher_concurrency,
        "teacher_tensor_parallel_size": teacher_tensor_parallel,
        "teacher_max_model_len": teacher_max_model_len,
        "max_prompt_chars": max_prompt_chars,
        "max_tokens_per_sample": max_tokens_per_sample,
        "teacher_context_reserve": teacher_context_reserve,
        "teacher_gpu_memory_utilization": teacher_gpu_memory_utilization,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "min_relative_val_improvement": min_relative_val_improvement,
    }, findings


def validate_dataset_config(path: Path, env: Mapping[str, str], target_tokens: int) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, [Finding("error", "dataset_config_missing", f"Dataset config does not exist: {path}")]
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [Finding("error", "dataset_config_invalid", f"Cannot parse {path}: {exc}")]

    mix = config.get("recommended_mix") or []
    try:
        mix_sum = sum(float(row["share"]) for row in mix)
    except (KeyError, TypeError, ValueError):
        mix_sum = -1.0
        findings.append(Finding("error", "invalid_dataset_mix", "Every recommended_mix entry needs a numeric share."))
    if mix_sum >= 0 and abs(mix_sum - 1.0) > 1e-6:
        findings.append(
            Finding(
                "error",
                "dataset_mix_not_normalized",
                f"configs/dataset_sources.json recommended_mix shares total {mix_sum:.4f}, not 1.0.",
            )
        )
    enabled = [row for row in config.get("sources", []) if row.get("enabled", False)]
    max_per_source = _integer(env, "DG_MAX_RECORDS_PER_SOURCE", 0, findings, 0)
    max_total = _integer(env, "DG_MAX_TOTAL_PROMPT_RECORDS", 0, findings, 0)
    max_tokens_per_sample = _integer(env, "DG_MAX_TOKENS_PER_SAMPLE", 4096, findings, 1)
    record_capacity = 0
    bad_sources: list[str] = []
    for source in enabled:
        try:
            source_capacity = int(source.get("max_records", 0))
        except (TypeError, ValueError):
            source_capacity = 0
        if source_capacity <= 0:
            bad_sources.append(str(source.get("id", "<unnamed>")))
            continue
        if max_per_source > 0:
            source_capacity = min(source_capacity, max_per_source)
        record_capacity += source_capacity
    if max_total > 0:
        record_capacity = min(record_capacity, max_total)
    token_capacity_upper_bound = record_capacity * max_tokens_per_sample

    if not enabled:
        findings.append(Finding("error", "no_enabled_sources", "Dataset config has no enabled sources."))
    if bad_sources:
        findings.append(Finding("error", "source_capacity_missing", f"Enabled sources need positive max_records: {', '.join(bad_sources)}"))
    if token_capacity_upper_bound < target_tokens:
        findings.append(
            Finding(
                "error",
                "source_capacity_too_small",
                f"Enabled-source upper bound is {token_capacity_upper_bound:,} teacher tokens "
                f"({record_capacity:,} records x {max_tokens_per_sample:,}), below the {target_tokens:,} target. "
                "Increase source max_records/add sources or lower the target.",
            )
        )
    elif token_capacity_upper_bound < int(target_tokens * 1.25):
        findings.append(
            Finding(
                "warning",
                "source_capacity_low_margin",
                "The theoretical source capacity is less than 25% above target; filtering or short answers can exhaust it early.",
            )
        )

    return {
        "path": str(path),
        "recommended_mix_sum": mix_sum,
        "enabled_sources": len(enabled),
        "record_capacity": record_capacity,
        "max_tokens_per_sample": max_tokens_per_sample,
        "teacher_token_capacity_upper_bound": token_capacity_upper_bound,
    }, findings


def estimate_disk_gib(settings: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, float]:
    # This is deliberately a conservative lower bound. Multimodal tensors can
    # dominate it, so users may raise DG_MIN_FREE_DISK_GB for media-heavy runs.
    target_tokens = int(settings["target_estimated_tokens"])
    blocks = int(settings["target_blocks"])
    prefix = int(settings["prefix_length"])
    canvas = int(settings["canvas_length"])
    retention = int(settings["checkpoint_retention"])
    teacher_gib = target_tokens * 6.0 / GIB
    corruption_gib = blocks * (prefix * 8 + canvas * 17 + 64) * 0.85 / GIB
    model_cache_gib = _number(env, "DG_MODEL_CACHE_ESTIMATE_GB", 80.0, [], 0.0)
    media_cache_gib = _number(env, "DG_MEDIA_CACHE_ESTIMATE_GB", 20.0, [], 0.0)
    adapter_checkpoint_gib = _number(env, "DG_ADAPTER_CHECKPOINT_ESTIMATE_GB", 4.0, [], 0.0)
    checkpoints_gib = adapter_checkpoint_gib * (retention + 1)
    # Include the standalone transplanted E4B base in the release bundle.
    working_gib = 32.0
    lower_bound = (
        teacher_gib
        + corruption_gib
        + model_cache_gib
        + media_cache_gib
        + checkpoints_gib
        + working_gib
    ) * 1.2
    explicit_min = _number(env, "DG_MIN_FREE_DISK_GB", 0.0, [], 0.0)
    return {
        "teacher_jsonl_gib": round(teacher_gib, 2),
        "corruption_lower_bound_gib": round(corruption_gib, 2),
        "model_cache_gib": round(model_cache_gib, 2),
        "media_cache_gib": round(media_cache_gib, 2),
        "checkpoint_budget_gib": round(checkpoints_gib, 2),
        "required_free_gib": round(max(lower_bound, explicit_min), 2),
    }


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(path)
    return candidate


def check_disk_locations(
    settings: Mapping[str, Any],
    env: Mapping[str, str],
) -> tuple[dict[str, Any], list[Finding]]:
    """Check the mounts that will actually hold Hub cache, data, and artifacts."""

    estimate = estimate_disk_gib(settings, env)
    hf_cache = Path(
        env.get("HF_HUB_CACHE")
        or (
            str(Path(env["HF_HOME"]).expanduser() / "hub")
            if env.get("HF_HOME")
            else str(Path.home() / ".cache" / "huggingface" / "hub")
        )
    )
    paths = {
        "hf_model_cache": hf_cache,
        "teacher_output": Path(
            env.get(
                "DG_TEACHER_OUTPUT",
                "data/teacher_supervised/teacher_outputs.jsonl",
            )
        ).parent,
        "media_cache": Path(env.get("DG_MEDIA_CACHE_DIR", "data/media_cache")),
        "corruption": Path(env.get("DG_CORRUPTION_DIR", "data/corruption")),
        "training": Path(
            env.get("DG_TRAIN_OUTPUT_DIR", "artifacts/conversion_training")
        ),
        "export": Path(
            env.get(
                "DG_EXPORT_OUTPUT",
                "artifacts/diffusiongemma-e4b-repro-bundle.tar.gz",
            )
        ).parent,
    }
    # Allocate the conservative estimates to their actual destinations. Working
    # space is split between training/export and temporary staging.
    requirements = {
        "hf_model_cache": estimate["model_cache_gib"] * 1.2,
        "teacher_output": estimate["teacher_jsonl_gib"] * 1.2,
        "media_cache": estimate["media_cache_gib"] * 1.2,
        "corruption": estimate["corruption_lower_bound_gib"] * 1.2,
        "training": (estimate["checkpoint_budget_gib"] + 6.0) * 1.2,
        # A replacement export can temporarily coexist with the previous
        # archive. Budget two incompressible E4B bases plus manifest overhead.
        "export": 40.0 * 1.2,
    }
    findings: list[Finding] = []
    mounts: dict[int, dict[str, Any]] = {}
    for role, requested_path in paths.items():
        try:
            existing = _nearest_existing_path(requested_path)
            device = int(existing.stat().st_dev)
            mount = mounts.setdefault(
                device,
                {
                    "device": device,
                    "check_path": str(existing),
                    "roles": [],
                    "paths": [],
                    "required_free_gib": 0.0,
                },
            )
            mount["roles"].append(role)
            mount["paths"].append(str(requested_path.expanduser().resolve(strict=False)))
            mount["required_free_gib"] += requirements[role]
        except OSError as exc:
            findings.append(
                Finding(
                    "error",
                    "disk_path_unavailable",
                    f"Cannot resolve storage path for {role}: {requested_path}: {exc}",
                )
            )

    explicit_min = float(estimate["required_free_gib"])
    allocated = sum(float(mount["required_free_gib"]) for mount in mounts.values())
    if explicit_min > allocated and mounts:
        training_mount = next(
            (
                mount
                for mount in mounts.values()
                if "training" in mount["roles"]
            ),
            next(iter(mounts.values())),
        )
        training_mount["required_free_gib"] += explicit_min - allocated
    reports: list[dict[str, Any]] = []
    for mount in mounts.values():
        free_gib = shutil.disk_usage(mount["check_path"]).free / GIB
        required = float(mount["required_free_gib"])
        row = {
            **mount,
            "required_free_gib": round(required, 2),
            "free_gib": round(free_gib, 2),
        }
        reports.append(row)
        if free_gib < required and env.get("DG_ALLOW_LOW_DISK") != "1":
            findings.append(
                Finding(
                    "error",
                    "insufficient_disk",
                    f"Storage for {', '.join(mount['roles'])} has {free_gib:.1f} GiB free "
                    f"at {mount['check_path']}; require {required:.1f} GiB. Move the actual "
                    "HF_HOME/DG_* output paths to larger volumes, or set DG_ALLOW_LOW_DISK=1 "
                    "only after independently sizing the run.",
                )
            )
    if env.get("DG_DISK_PATH"):
        findings.append(
            Finding(
                "warning",
                "disk_path_is_check_only",
                "DG_DISK_PATH does not redirect Hugging Face or pipeline outputs. Set HF_HOME, "
                "DG_CORRUPTION_DIR, and DG_TRAIN_OUTPUT_DIR to choose real storage locations.",
            )
        )
    return {"estimate": estimate, "mounts": reports}, findings


def _teacher_urls(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    return [base + "/models", root + "/health"]


def probe_teacher(
    base_url: str,
    api_key: str | None = None,
    expected_model: str | None = None,
) -> dict[str, Any]:
    import requests

    statuses: dict[str, Any] = {}
    ready = False
    served_models: list[str] = []
    model_url = _teacher_urls(base_url)[0]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    for url in _teacher_urls(base_url):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            statuses[url] = response.status_code
            if url == model_url and 200 <= response.status_code < 300:
                try:
                    payload = response.json()
                    rows = payload.get("data") if isinstance(payload, dict) else None
                    if isinstance(rows, list):
                        served_models = [
                            str(row.get("id")) for row in rows
                            if isinstance(row, dict) and row.get("id") is not None
                        ]
                        ready = bool(served_models) and (
                            expected_model is None or expected_model in served_models
                        )
                except ValueError:
                    ready = False
        except requests.RequestException as exc:
            statuses[url] = type(exc).__name__
    return {"ready": ready, "statuses": statuses, "served_models": served_models}


def _is_local_host(host: str | None) -> bool:
    return (host or "").lower() in {"127.0.0.1", "localhost", "0.0.0.0", "::1", "::"}


def local_port_available(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    family = socket.AF_INET6 if host in {"::1", "::"} else socket.AF_INET
    bind_host = "::1" if family == socket.AF_INET6 else "127.0.0.1"
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
        return True
    except OSError:
        return False


def check_imports(train_mode: str, require_vllm: bool) -> tuple[dict[str, str], list[Finding]]:
    findings: list[Finding] = []
    modules = list(REQUIRED_IMPORTS)
    if train_mode in {"lora", "qlora"}:
        modules.append("peft")
    if train_mode == "qlora":
        modules.append("bitsandbytes")
    if require_vllm:
        modules.append("vllm")
    versions: dict[str, str] = {}
    for name in modules:
        if importlib.util.find_spec(name) is None:
            findings.append(Finding("error", "missing_import", f"Missing Python package {name!r}; rerun scripts/linux/setup.sh."))
            continue
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:  # noqa: BLE001
            findings.append(Finding("error", "broken_import", f"Importing {name!r} failed: {type(exc).__name__}: {exc}"))
    try:
        transformers_module = importlib.import_module("transformers")
        if not hasattr(transformers_module, "DiffusionGemmaForBlockDiffusion"):
            findings.append(
                Finding(
                    "error",
                    "diffusiongemma_loader_missing",
                    "Installed Transformers lacks DiffusionGemmaForBlockDiffusion; install the version constrained by pyproject.toml.",
                )
            )
    except Exception:
        pass
    return versions, findings


def check_hf_models(env: Mapping[str, str]) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    results: dict[str, Any] = {}
    try:
        from huggingface_hub import get_token
        from transformers import AutoConfig
    except Exception as exc:  # noqa: BLE001
        return results, [Finding("error", "hf_import_failed", f"Cannot import Hugging Face clients: {exc}")]

    token = get_token()
    teacher = env.get("DG_MODEL", "google/gemma-4-E4B-it")
    student_source = env.get("DG_STUDENT_MODEL", teacher)
    models: list[tuple[str, str, str | None]] = [
        ("teacher", teacher, "gemma4")
    ]
    if student_source != teacher:
        models.append(("student_source", student_source, "gemma4"))

    for role, model_id, expected_type in models:
        is_local = Path(model_id).exists()
        if model_id.startswith("google/") and not token and not is_local:
            findings.append(
                Finding(
                    "error",
                    "hf_token_missing",
                    f"{role.title()} model {model_id} is gated. Accept its Hugging Face terms and run `hf auth login` before the pipeline.",
                )
            )
            continue
        try:
            config = AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=True)
            model_type = str(getattr(config, "model_type", "unknown"))
            results[role] = {"model": model_id, "model_type": model_type, "config_access": True}
            if expected_type and model_type != expected_type:
                findings.append(
                    Finding(
                        "error",
                        "unexpected_model_type",
                        f"{role.title()} {model_id} has model_type={model_type!r}; expected {expected_type!r}.",
                    )
                )
            text_config = getattr(config, "text_config", None)
            if role in {"teacher", "student_source"} and text_config is not None:
                expected_e4b = {
                    "hidden_size": 2560,
                    "intermediate_size": 10240,
                    "num_hidden_layers": 42,
                    "hidden_size_per_layer_input": 256,
                    "vocab_size_per_layer_input": 262144,
                    "num_kv_shared_layers": 18,
                    "enable_moe_block": False,
                }
                mismatches = {
                    key: {
                        "expected": expected,
                        "actual": getattr(text_config, key, None),
                    }
                    for key, expected in expected_e4b.items()
                    if getattr(text_config, key, None) != expected
                }
                results[role]["e4b_contract"] = {
                    "ok": not mismatches,
                    "mismatches": mismatches,
                }
                if mismatches:
                    findings.append(
                        Finding(
                            "error",
                            "unexpected_e4b_architecture",
                            f"{role.title()} {model_id} does not match the Gemma 4 E4B "
                            f"architecture: {mismatches}.",
                        )
                    )
                missing_modalities = [
                    name
                    for name in ("vision_config", "audio_config")
                    if getattr(config, name, None) is None
                ]
                results[role]["multimodal_contract"] = {
                    "ok": not missing_modalities,
                    "missing": missing_modalities,
                }
                if missing_modalities:
                    findings.append(
                        Finding(
                            "error",
                            "e4b_multimodal_tower_missing",
                            f"{role.title()} {model_id} is missing required "
                            f"Gemma 4 E4B modalities: {missing_modalities}.",
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            findings.append(
                Finding(
                    "error",
                    "hf_config_unavailable",
                    f"Cannot access {role} config for {model_id}: {type(exc).__name__}: {str(exc)[:300]}",
                )
            )
    return results, findings


def check_hf_dataset_sources(
    dataset_config: Path,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    """Verify current Hub config/split contracts before opening a long stream."""

    findings: list[Finding] = []
    results: list[dict[str, Any]] = []
    try:
        config = json.loads(dataset_config.read_text(encoding="utf-8"))
        from datasets import get_dataset_config_names, get_dataset_split_names
        from huggingface_hub import get_token
    except Exception as exc:  # noqa: BLE001
        return results, [
            Finding(
                "error",
                "dataset_access_check_unavailable",
                f"Cannot initialize dataset Hub checks: {type(exc).__name__}: {exc}",
            )
        ]
    token = get_token()
    for source in config.get("sources", []):
        if not source.get("enabled", True) or source.get("manifest_only"):
            continue
        dataset_id = str(source.get("id") or "")
        requested_config = source.get("config")
        requested_split = str(source.get("split", "train"))
        try:
            configs = list(get_dataset_config_names(dataset_id, token=token))
            if requested_config is not None:
                selected_config = str(requested_config)
                if selected_config not in configs:
                    raise ValueError(
                        f"config {selected_config!r} not in current configs {configs[:20]}"
                    )
            elif "default" in configs:
                selected_config = "default"
            elif len(configs) == 1:
                selected_config = str(configs[0])
            else:
                raise ValueError(
                    f"dataset has multiple configs and none was selected: {configs[:20]}"
                )
            splits = list(
                get_dataset_split_names(
                    dataset_id,
                    selected_config,
                    token=token,
                )
            )
            if requested_split not in splits:
                raise ValueError(
                    f"split {requested_split!r} not in current splits {splits}"
                )
            results.append(
                {
                    "id": dataset_id,
                    "config": selected_config,
                    "split": requested_split,
                    "accessible": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "id": dataset_id,
                    "config": requested_config,
                    "split": requested_split,
                    "accessible": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            )
            findings.append(
                Finding(
                    "error",
                    "dataset_source_unavailable",
                    f"Cannot access configured dataset {dataset_id} "
                    f"(config={requested_config!r}, split={requested_split!r}): "
                    f"{type(exc).__name__}: {str(exc)[:300]}. Accept gated dataset "
                    "terms when applicable or update/disable the source before the run.",
                )
            )
    return results, findings


def check_gpu(preset: str, env: Mapping[str, str]) -> tuple[dict[str, Any], list[Finding]]:
    if env.get("DG_SKIP_GPU_PREFLIGHT") == "1":
        return {"check_explicitly_skipped": True}, [
            Finding(
                "warning",
                "gpu_check_skipped",
                "GPU checks were explicitly skipped; both gpu and smoke presets still load and transplant Gemma 4 E4B.",
            )
        ]
    findings: list[Finding] = []
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {}, [Finding("error", "torch_import_failed", f"Cannot inspect CUDA because torch import failed: {exc}")]
    if not torch.cuda.is_available():
        return {"cuda_available": False}, [Finding("error", "cuda_unavailable", "CUDA is unavailable. Run the gpu preset on a CUDA cloud instance.")]
    bf16 = bool(torch.cuda.is_bf16_supported())
    devices = []
    total_gib = 0.0
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        memory_gib = props.total_memory / GIB
        total_gib += memory_gib
        devices.append({"index": index, "name": props.name, "memory_gib": round(memory_gib, 2), "compute_capability": f"{props.major}.{props.minor}"})
    teacher_tp = _integer(
        env, "DG_TEACHER_TENSOR_PARALLEL_SIZE", 1, findings, 1
    )
    if teacher_tp > len(devices):
        findings.append(
            Finding(
                "error",
                "teacher_tensor_parallel_exceeds_visible_gpus",
                f"DG_TEACHER_TENSOR_PARALLEL_SIZE={teacher_tp}, but only {len(devices)} CUDA device(s) are visible.",
            )
        )
    if not bf16:
        findings.append(Finding("error", "bf16_unsupported", "The production preset requires a CUDA GPU with native BF16 support."))
    min_gpu_gib = _number(env, "DG_MIN_GPU_GB", 75.0, findings, 0.0)
    free_gib = 0.0
    for index, device in enumerate(devices):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
            device["free_memory_gib"] = round(free_bytes / GIB, 2)
            free_gib += free_bytes / GIB
        except Exception:  # noqa: BLE001
            device["free_memory_gib"] = None
    if total_gib < min_gpu_gib and env.get("DG_ALLOW_LOW_VRAM") != "1":
        findings.append(
            Finding(
                "error",
                "insufficient_vram",
                f"Visible GPU memory totals {total_gib:.1f} GiB; require at least {min_gpu_gib:.1f} GiB for this QLoRA configuration. "
                "Use a larger/multi-GPU instance, or explicitly set DG_ALLOW_LOW_VRAM=1 after sizing the run.",
            )
        )
    min_free_gib = _number(env, "DG_MIN_FREE_GPU_GB", 70.0, findings, 0.0)
    if free_gib and free_gib < min_free_gib and env.get("DG_ALLOW_LOW_VRAM") != "1":
        findings.append(
            Finding(
                "error",
                "insufficient_free_vram",
                f"Visible GPUs have only {free_gib:.1f} GiB free; require {min_free_gib:.1f} GiB "
                "before starting the teacher/student lifecycle.",
            )
        )
    return {
        "cuda_available": True,
        "bf16_supported": bf16,
        "total_memory_gib": round(total_gib, 2),
        "free_memory_gib": round(free_gib, 2),
        "devices": devices,
    }, findings


def check_host_memory(env: Mapping[str, str]) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            fields[key] = int(value.strip().split()[0]) * 1024
        total_gib = fields["MemTotal"] / GIB
        available_gib = fields.get("MemAvailable", fields["MemTotal"]) / GIB
    except (OSError, KeyError, ValueError):
        return {"available": False}, [
            Finding("warning", "host_memory_unknown", "Could not read /proc/meminfo; host RAM was not verified.")
        ]
    minimum_gib = _number(env, "DG_MIN_HOST_RAM_GB", 64.0, findings, 0.0)
    if total_gib < minimum_gib and env.get("DG_ALLOW_LOW_HOST_RAM") != "1":
        findings.append(
            Finding(
                "error",
                "insufficient_host_ram",
                f"Host RAM is {total_gib:.1f} GiB; require at least {minimum_gib:.1f} GiB for model loading, data workers, and export. "
                "Use a larger instance or explicitly set DG_ALLOW_LOW_HOST_RAM=1 after profiling.",
            )
        )
    minimum_available_gib = _number(env, "DG_MIN_AVAILABLE_HOST_RAM_GB", 32.0, findings, 0.0)
    if available_gib < minimum_available_gib and env.get("DG_ALLOW_LOW_HOST_RAM") != "1":
        findings.append(
            Finding(
                "error",
                "insufficient_available_host_ram",
                f"Only {available_gib:.1f} GiB host RAM is currently available; require "
                f"{minimum_available_gib:.1f} GiB before model loading.",
            )
        )
    return {
        "total_gib": round(total_gib, 2),
        "available_gib": round(available_gib, 2),
        "minimum_gib": minimum_gib,
        "minimum_available_gib": minimum_available_gib,
    }, findings


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_preflight(preset: str, dataset_config: Path, output: Path, skip_hf_check: bool = False) -> dict[str, Any]:
    env = dict(os.environ)
    findings: list[Finding] = []
    if sys.version_info < (3, 11):
        findings.append(Finding("error", "python_too_old", f"Python 3.11+ is required; found {sys.version.split()[0]}."))

    settings, numeric_findings = validate_numeric_settings(env)
    findings.extend(numeric_findings)
    dataset, dataset_findings = validate_dataset_config(dataset_config, env, int(settings["target_estimated_tokens"]))
    findings.extend(dataset_findings)

    student_init = env.get("DG_STUDENT_INIT", "transplant")
    if student_init != "transplant":
        findings.append(
            Finding(
                "error",
                "invalid_student_init",
                "Production requires DG_STUDENT_INIT=transplant so the final model "
                "remains an E4B-sized Diffusion Transformer.",
            )
        )

    base_url = env.get("DG_TEACHER_BASE_URL", f"http://127.0.0.1:{env.get('DG_TEACHER_PORT', '8000')}/v1")
    try:
        teacher = probe_teacher(
            base_url,
            api_key=env.get("DG_TEACHER_API_KEY"),
            expected_model=env.get("DG_TEACHER_SERVED_MODEL_NAME", env.get("DG_MODEL", "google/gemma-4-E4B-it")),
        )
    except Exception as exc:  # noqa: BLE001
        teacher = {"ready": False, "statuses": {}, "probe_error": f"{type(exc).__name__}: {exc}"}
    parsed = urlparse(base_url)
    require_vllm = False
    if teacher["ready"]:
        teacher["mode"] = "reuse_existing"
        if _is_local_host(parsed.hostname) and env.get("DG_ALLOW_REUSE_LOCAL_TEACHER") != "1":
            findings.append(
                Finding(
                    "error",
                    "local_teacher_gpu_ownership_unknown",
                    "A healthy local teacher is already running. The pipeline will not kill user-owned processes, "
                    "so training could OOM if it shares the student GPUs. Stop it and let the pipeline auto-start/stop it, "
                    "or set DG_ALLOW_REUSE_LOCAL_TEACHER=1 only when it runs on separate GPUs.",
                )
            )
    elif _is_local_host(parsed.hostname):
        if not local_port_available(base_url):
            findings.append(
                Finding(
                    "error",
                    "teacher_port_collision",
                    f"Teacher endpoint is not healthy (only HTTP 2xx counts as ready), but port {parsed.port or 80} is already occupied. Stop the conflicting process or set DG_TEACHER_PORT.",
                )
            )
        else:
            require_vllm = True
            teacher["mode"] = "auto_start"
    else:
        findings.append(Finding("error", "remote_teacher_unavailable", f"Configured remote teacher endpoint is not ready: {base_url}"))

    versions, import_findings = check_imports(env.get("DG_TRAIN_MODE", "qlora"), require_vllm=require_vllm)
    findings.extend(import_findings)
    if skip_hf_check:
        models: dict[str, Any] = {"check_skipped": True}
        dataset_sources: list[dict[str, Any]] = []
        findings.append(Finding("warning", "hf_check_skipped", "Hugging Face gated-config access was not checked."))
    else:
        models, hf_findings = check_hf_models(env)
        findings.extend(hf_findings)
        dataset_sources, dataset_access_findings = check_hf_dataset_sources(
            dataset_config
        )
        findings.extend(dataset_access_findings)

    gpu, gpu_findings = check_gpu(preset, env)
    findings.extend(gpu_findings)
    host_memory, host_memory_findings = check_host_memory(env)
    findings.extend(host_memory_findings)

    disk, disk_findings = check_disk_locations(settings, env)
    findings.extend(disk_findings)

    errors = [asdict(item) for item in findings if item.level == "error"]
    warnings = [asdict(item) for item in findings if item.level == "warning"]
    report = {
        "ok": not errors,
        "preset": preset,
        "student_init": student_init,
        "settings": settings,
        "dataset": dataset,
        "teacher": teacher,
        "models": models,
        "dataset_sources": dataset_sources,
        "python": {"version": sys.version.split()[0], "imports": versions},
        "gpu": gpu,
        "host_memory": host_memory,
        "disk": disk,
        "errors": errors,
        "warnings": warnings,
    }
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast cloud pipeline readiness checks")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=os.environ.get("DG_PRESET", "gpu"))
    parser.add_argument("--dataset-config", type=Path, default=Path(os.environ.get("DG_DATASET_CONFIG", "configs/dataset_sources.json")))
    parser.add_argument("--output", type=Path, default=Path("outputs/logs/preflight.json"))
    parser.add_argument("--skip-hf-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = run_preflight(args.preset, args.dataset_config, args.output, skip_hf_check=args.skip_hf_check)
    if report["ok"]:
        print(f"Preflight passed ({len(report['warnings'])} warning(s)); report: {args.output}")
        return
    print(f"Preflight failed with {len(report['errors'])} error(s). Report: {args.output}", file=sys.stderr)
    for item in report["errors"]:
        print(f"  - [{item['code']}] {item['message']}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
