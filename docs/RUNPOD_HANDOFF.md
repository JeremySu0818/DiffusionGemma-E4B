# RunPod-First Cloud Handoff

This handoff is the required route when the local probe reports `runpod_first_cloud_handoff_required`.

## 1. Instance Choice

Recommended primary pod:

- GPU: NVIDIA A100 80GB PCIe or H100 80GB
- VRAM: 80GB
- RAM: 120GB minimum, 180GB preferred
- Disk: 300GB container disk plus 500GB network volume
- Image/template: RunPod PyTorch 2.8+ CUDA 12.8 Ubuntu template, or `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- Network volume: yes, mount at `/workspace`
- Estimated cost: A100 80GB often around 1.8-2.8 USD/hour, H100 80GB often around 3.5-5.5 USD/hour depending on market
- Estimated time: generation 6-18 hours depending throughput; corruption under 1 hour; QLoRA conversion training 18-48 hours for the configured 200k-step run; full BF16 training requires multi-GPU and is not the first-pass recommendation

Fallback pod:

- GPU: RTX 6000 Ada 48GB or L40S 48GB
- VRAM: 48GB
- RAM: 96GB minimum
- Disk: 300GB container disk plus 500GB network volume
- Training mode: QLoRA only, batch size 1, gradient accumulation 64 if OOM

## 2. RunPod Web UI Steps

1. Open RunPod console.
2. Choose Secure Cloud if available for more stable long jobs; Community Cloud is acceptable if price is the priority.
3. Select GPU `A100 80GB` first, then `H100 80GB` if A100 is unavailable.
4. Select template `RunPod PyTorch` with Python 3.11 and CUDA 12.8. If there is a text box for Docker image, use:

```bash
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

5. Container disk: 300GB.
6. Network volume: create or attach 500GB at `/workspace`.
7. Expose ports:
   - `22/tcp` for SSH
   - `6006/http` optional TensorBoard
   - `8000/http` optional vLLM teacher endpoint
8. Start pod.
9. Open SSH from RunPod UI and paste the SSH command into a local terminal.

## 3. One-Command Environment Setup

Set the repo source. If this local folder is not pushed to GitHub, upload it with `rsync` or RunPod web upload first.

```bash
export DG_REPO_URL="https://github.com/YOUR_ACCOUNT/DiffusionGemma-E4B.git"
cd /workspace
git clone "$DG_REPO_URL" DiffusionGemma-E4B
cd /workspace/DiffusionGemma-E4B
bash scripts/runpod_setup.sh
```

If uploading from Windows instead of Git:

```powershell
scp -r C:\AI\DiffusionGemma-E4B root@RUNPOD_HOST:/workspace/DiffusionGemma-E4B
```

Then on RunPod:

```bash
cd /workspace/DiffusionGemma-E4B
bash scripts/runpod_setup.sh
```

## 4. Hugging Face Login And Weight Cache

```bash
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
huggingface-cli login
export HF_HOME=/workspace/hf-cache
export TRANSFORMERS_CACHE=/workspace/hf-cache
mkdir -p "$HF_HOME"
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
for model in ["google/gemma-4-E4B-it", "google/diffusiongemma-26B-A4B-it"]:
    print("checking", model)
    AutoConfig.from_pretrained(model)
AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
PY
df -h /workspace
```

If login or license fails:

```bash
huggingface-cli whoami
huggingface-cli logout
huggingface-cli login
python - <<'PY'
from huggingface_hub import model_info
print(model_info("google/gemma-4-E4B-it"))
PY
```

Accept the Gemma license in the browser under the model card, then rerun the cache check.

## 5. Teacher Runtime

Use vLLM as a local OpenAI-compatible teacher endpoint on the same pod:

```bash
source .venv/bin/activate
pip install "vllm>=0.10.0"
tmux new -s teacher
vllm serve google/gemma-4-E4B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

Detach with `Ctrl-b d`. Confirm:

```bash
curl http://127.0.0.1:8000/v1/models
```

## 6. Prompt-Free Self-Continuation Generation

```bash
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
export DG_TEACHER_BASE_URL=http://127.0.0.1:8000/v1
tmux new -s generate
bash scripts/runpod_generate.sh 2>&1 | tee outputs/logs/generate.log
```

Resume after interruption:

```bash
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
export DG_TEACHER_BASE_URL=http://127.0.0.1:8000/v1
bash scripts/runpod_generate.sh 2>&1 | tee -a outputs/logs/generate.log
```

Confirm >= 50,000,000 estimated tokens:

```bash
cat data/raw_self_continuation/progress.json
python - <<'PY'
import json
p=json.load(open("data/raw_self_continuation/progress.json"))
assert p["estimated_tokens"] >= 50000000, p
print(p)
PY
```

## 7. Diffusion Corruption Shards

```bash
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
bash scripts/runpod_corrupt.sh 2>&1 | tee outputs/logs/corrupt.log
python -m diffusiongemma_e4b.corruption --output-dir data/corruption --target-blocks 200000 --verify-only
```

The shards contain:

- `prefix_ids`: causal prefix, left padded
- `prefix_lens`: real prefix lengths
- `target_ids`: clean 256-token target block
- `corrupted_ids`: noisy canvas
- `corruption_masks`: positions replaced by random vocabulary tokens
- `noise_t`: corruption level

## 8. Conversion Training

First pass uses QLoRA:

```bash
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
tmux new -s train
bash scripts/runpod_transplant_train.sh 2>&1 | tee outputs/logs/train.log
```

Training settings:

- Mode: QLoRA
- Batch size: 1
- Gradient accumulation: 32
- Sequence: prefix 512 + canvas 256
- LR: `2e-4`
- Optimizer: AdamW
- dtype: bf16
- Save interval: 1000 steps
- Validation interval: 500 steps
- Target steps: 200,000
- Self-conditioning probability: 0.5

Resume:

```bash
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
bash scripts/runpod_transplant_train.sh 2>&1 | tee -a outputs/logs/train.log
```

## 9. Monitoring

```bash
cd /workspace/DiffusionGemma-E4B
bash scripts/runpod_monitor.sh
```

Manual checks:

```bash
nvidia-smi
df -h /workspace
tail -f outputs/logs/train.log
tail -f artifacts/conversion_training/train_log.jsonl
find artifacts/conversion_training -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail
```

Optional TensorBoard:

```bash
source .venv/bin/activate
pip install tensorboard
tensorboard --logdir artifacts/conversion_training --host 0.0.0.0 --port 6006
```

## 10. Validation

```bash
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
bash scripts/runpod_validate_export.sh 2>&1 | tee outputs/logs/validate_export.log
```

Explicit checks:

```bash
python -m diffusiongemma_e4b.validate \
  --model-dir artifacts/conversion_training/final \
  --data-dir data/corruption \
  --output outputs/validation/validation_report.json

python -m diffusiongemma_e4b.infer \
  --model-dir artifacts/conversion_training/final \
  --prompt "" \
  --output outputs/validation/strict_diffusion_inference.json \
  --max-new-tokens 256 \
  --denoise-steps 32 \
  --entropy-bound 0.1
```

Fake-wrapper exclusion:

```bash
python - <<'PY'
from pathlib import Path
src=Path("src/diffusiongemma_e4b/infer.py").read_text()
assert ".generate(" not in src
assert '"ar_fallback_used": False' in src
print("strict diffusion inference path does not use HF generate")
PY
```

## 11. Export And Download

```bash
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
python -m diffusiongemma_e4b.export --output artifacts/diffusiongemma-e4b-repro-bundle.tar.gz
ls -lh artifacts/diffusiongemma-e4b-repro-bundle.tar.gz*
```

Download to Windows:

```powershell
scp root@RUNPOD_HOST:/workspace/DiffusionGemma-E4B/artifacts/diffusiongemma-e4b-repro-bundle.tar.gz C:\AI\DiffusionGemma-E4B\artifacts\
scp root@RUNPOD_HOST:/workspace/DiffusionGemma-E4B/artifacts/diffusiongemma-e4b-repro-bundle.tar.gz.json C:\AI\DiffusionGemma-E4B\artifacts\
```

## 12. Failure Recovery

OOM:

```bash
python -m diffusiongemma_e4b.train \
  --model-dir artifacts/transplanted \
  --data-dir data/corruption \
  --output-dir artifacts/conversion_training \
  --train-mode qlora \
  --batch-size 1 \
  --gradient-accumulation-steps 64 \
  --learning-rate 1e-4 \
  --max-steps 200000 \
  --save-interval 500 \
  --val-interval 500 \
  --gradient-checkpointing \
  --resume
```

Disk full:

```bash
df -h /workspace
find artifacts/conversion_training -maxdepth 1 -type d -name 'checkpoint-*' | sort | head
tar -czf /workspace/old-checkpoints.tar.gz artifacts/conversion_training/checkpoint-0000*
rm -rf artifacts/conversion_training/checkpoint-0000*
```

HF download failure:

```bash
export HF_HOME=/workspace/hf-cache
huggingface-cli login
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("google/gemma-4-E4B-it", local_dir="/workspace/hf-cache/gemma-4-E4B-it")
PY
```

Interrupted generation:

```bash
bash scripts/runpod_generate.sh 2>&1 | tee -a outputs/logs/generate.log
```

Interrupted training:

```bash
bash scripts/runpod_transplant_train.sh 2>&1 | tee -a outputs/logs/train.log
```

NaN or bad loss:

```bash
python -m diffusiongemma_e4b.train \
  --model-dir artifacts/transplanted \
  --data-dir data/corruption \
  --output-dir artifacts/conversion_training_recovery \
  --train-mode qlora \
  --batch-size 1 \
  --gradient-accumulation-steps 64 \
  --learning-rate 5e-5 \
  --max-grad-norm 0.5 \
  --max-steps 200000 \
  --self-conditioning-prob 0.25 \
  --gradient-checkpointing \
  --resume
```

RunPod preemption:

```bash
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
cat artifacts/conversion_training/latest_checkpoint.txt
bash scripts/runpod_transplant_train.sh 2>&1 | tee -a outputs/logs/train_after_preempt.log
```

First command to run after local infeasibility is confirmed:

```bash
cd /workspace/DiffusionGemma-E4B && bash scripts/runpod_setup.sh
```
