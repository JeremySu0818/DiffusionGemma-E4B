#!/usr/bin/env bash
set -euo pipefail
cd /workspace/DiffusionGemma-E4B
watch -n 5 'nvidia-smi; echo; df -h /workspace; echo; tail -n 20 artifacts/conversion_training/train_log.jsonl 2>/dev/null || true; echo; find artifacts/conversion_training -maxdepth 1 -type d -name "checkpoint-*" | sort | tail'
