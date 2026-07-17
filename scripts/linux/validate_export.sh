#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate

TRAIN_OUTPUT_DIR="${DG_TRAIN_OUTPUT_DIR:-artifacts/conversion_training}"
FINAL_MODEL_DIR="$TRAIN_OUTPUT_DIR/final"
if [[ -n "${DG_FINAL_MODEL_DIR:-}" && "$DG_FINAL_MODEL_DIR" != "$FINAL_MODEL_DIR" ]]; then
  echo "DG_FINAL_MODEL_DIR must equal $FINAL_MODEL_DIR." >&2
  echo "Set DG_TRAIN_OUTPUT_DIR to relocate training and final artifacts together." >&2
  exit 2
fi
CORRUPTION_DIR="${DG_CORRUPTION_DIR:-data/corruption}"
VALIDATION_DIR="${DG_VALIDATION_DIR:-outputs/validation}"
VALIDATION_REPORT="$VALIDATION_DIR/validation_report.json"
INFERENCE_REPORT="$VALIDATION_DIR/strict_diffusion_inference.json"
BUNDLE_OUTPUT="${DG_EXPORT_OUTPUT:-artifacts/diffusiongemma-e4b-repro-bundle.tar.gz}"
mkdir -p "$VALIDATION_DIR"
# Reports are run-scoped: never let a skipped/failed validation export stale JSON.
rm -f "$VALIDATION_REPORT" "$INFERENCE_REPORT"

validate_args=(
  --model-dir "$FINAL_MODEL_DIR"
  --data-dir "$CORRUPTION_DIR"
  --output "$VALIDATION_REPORT"
  --dtype "${DG_DTYPE:-bfloat16}"
  --device-map "${DG_DEVICE_MAP:-auto}"
)
if [[ "${DG_TRAIN_MODE:-qlora}" == "qlora" ]]; then
  validate_args+=(--load-in-4bit)
fi
if [[ -n "${DG_PROCESSOR_PATH:-}" ]]; then
  validate_args+=(--processor "$DG_PROCESSOR_PATH")
fi
if [[ -n "${DG_ADAPTER_BASE_MODEL:-}" ]]; then
  validate_args+=(--base-model "$DG_ADAPTER_BASE_MODEL")
fi
if [[ "${DG_SKIP_FORWARD_VALIDATION:-0}" == "1" ]]; then
  validate_args+=(--skip-forward)
fi
python -m diffusiongemma_e4b.validate "${validate_args[@]}"

if [[ "${DG_SKIP_INFERENCE_VALIDATION:-0}" != "1" ]]; then
  max_new_tokens="${DG_VALIDATION_MAX_NEW_TOKENS:-256}"
  if [[ "${DG_PRESET:-gpu}" == "smoke" && -z "${DG_VALIDATION_MAX_NEW_TOKENS+x}" ]]; then
    max_new_tokens=32
  fi
  infer_args=(
    --model-dir "$FINAL_MODEL_DIR"
    --prompt "${DG_VALIDATION_PROMPT:-Explain why the sky appears blue in three concise sentences.}"
    --output "$INFERENCE_REPORT"
    --max-new-tokens "$max_new_tokens"
    --denoise-steps "${DG_DENOISE_STEPS:-48}"
    --entropy-bound "${DG_ENTROPY_BOUND:-0.1}"
    --confidence-threshold "${DG_CONFIDENCE_THRESHOLD:-0.005}"
    --stability-steps "${DG_STABILITY_STEPS:-1}"
    --t-min "${DG_T_MIN:-0.4}"
    --t-max "${DG_T_MAX:-0.8}"
    --dtype "${DG_DTYPE:-bfloat16}"
    --device-map "${DG_DEVICE_MAP:-auto}"
    --seed "${DG_SEED:-1337}"
  )
  if [[ "${DG_TRAIN_MODE:-qlora}" == "qlora" ]]; then
    infer_args+=(--load-in-4bit)
  fi
  if [[ -n "${DG_PROCESSOR_PATH:-}" ]]; then
    infer_args+=(--processor "$DG_PROCESSOR_PATH")
  fi
  if [[ -n "${DG_ADAPTER_BASE_MODEL:-}" ]]; then
    infer_args+=(--base-model "$DG_ADAPTER_BASE_MODEL")
  fi
  python -m diffusiongemma_e4b.infer "${infer_args[@]}"
  python - "$INFERENCE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("strict_diffusion") is not True or payload.get("ar_fallback_used") is not False:
    raise SystemExit("inference validation did not use strict DiffusionGemma generation")
if not str(payload.get("text") or "").strip():
    raise SystemExit("inference validation produced empty text")
PY
fi

python -m diffusiongemma_e4b.export \
  --output "$BUNDLE_OUTPUT" \
  --model-dir "$FINAL_MODEL_DIR" \
  --validation-dir "$VALIDATION_DIR" \
  --base-model-dir "${DG_TRANSPLANT_DIR:-artifacts/transplanted}"
