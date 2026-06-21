#!/usr/bin/env bash
set -euo pipefail

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

watch -n 5 'nvidia-smi; echo; df -h .; echo; tail -n 20 artifacts/conversion_training/train_log.jsonl 2>/dev/null || true; echo; find artifacts/conversion_training -maxdepth 1 -type d -name "checkpoint-*" | sort | tail'
