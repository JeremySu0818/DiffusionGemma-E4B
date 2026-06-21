# Cloud Deployment Guide (Linux/Unix & Windows)

This guide provides instructions for deploying and reproducing the DiffusionGemma-E4B conversion pipeline on a cloud GPU virtual machine (e.g., RunPod, Lambda Labs, Vast.ai, or custom SSH-enabled Linux instances) as well as running local tests on Windows.

---

## 1. Cloud Instance Selection

When selecting a cloud GPU instance for training, use these specifications:

- **Primary Recommendation:**
  - **GPU:** NVIDIA A100 80GB PCIe or H100 80GB (VRAM: 80GB)
  - **RAM:** 120GB minimum, 180GB preferred
  - **Disk Space:** 300GB container/OS disk, plus 500GB persistent volume
  - **Estimated Time:** 
    - Teacher Generation: 6–18 hours
    - Corruption Data Generation: < 1 hour
    - QLoRA Training: 18–48 hours (for 200k steps)
- **Secondary / Cost-Effective Recommendation:**
  - **GPU:** RTX 6000 Ada 48GB or L40S 48GB (VRAM: 48GB)
  - **RAM:** 96GB minimum
  - **Training Mode:** QLoRA only, batch size 1, gradient accumulation steps 64 (to prevent Out-Of-Memory)

---

## 2. Cloud Server Environment Setup

Prepare your environment by cloning the repository and running the setup script.

```bash
# Set your repository URL (if cloned via Git)
export DG_REPO_URL="https://github.com/YOUR_ACCOUNT/DiffusionGemma-E4B.git"
git clone "$DG_REPO_URL" DiffusionGemma-E4B

cd DiffusionGemma-E4B
bash scripts/linux/setup.sh
```

If you are uploading files directly via `rsync` or `scp` from your local machine:

```bash
cd DiffusionGemma-E4B
bash scripts/linux/setup.sh
```

---

## 3. Hugging Face Login and Weights Caching

Log in to Hugging Face to download the Gemma-4 models.

```bash
cd DiffusionGemma-E4B
source .venv/bin/activate
huggingface-cli login

# Setup Cache Location
export HF_HOME=$(pwd)/.hf-cache
export TRANSFORMERS_CACHE=$(pwd)/.hf-cache
mkdir -p "$HF_HOME"

# Verify model access and cache download
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
for model in ["google/gemma-4-E4B-it", "google/diffusiongemma-26B-A4B-it"]:
    print("Checking access for:", model)
    AutoConfig.from_pretrained(model)
AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
PY
```

> [!NOTE]
> Make sure to accept the Gemma license on the Hugging Face model cards before attempting to access the weights.

---

## 4. Teacher Runtime (Cloud GPU VM)

To run the teacher model efficiently, start a vLLM server to expose an OpenAI-compatible API:

```bash
source .venv/bin/activate
pip install "vllm>=0.10.0"

# Run vLLM inside tmux to keep it running in the background
tmux new -s teacher
vllm serve google/gemma-4-E4B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```
*(Press `Ctrl-b` then `d` to detach from the tmux session).*

Confirm the server is healthy:
```bash
curl http://127.0.0.1:8000/v1/models
```

---

## 5. Running the Pipeline (Linux/Shell)

The pipeline is split into three main phases: generation, data corruption, training, and validation.

### Step 1: Teacher Dataset Generation
```bash
cd DiffusionGemma-E4B
source .venv/bin/activate
export DG_TEACHER_BASE_URL=http://127.0.0.1:8000/v1

# Run generation in a tmux session
tmux new -s generate
bash scripts/linux/generate.sh 2>&1 | tee outputs/logs/generate.log
```

### Step 2: Block Corruption Generation
```bash
bash scripts/linux/corrupt.sh 2>&1 | tee outputs/logs/corrupt.log
```

### Step 3: Weight Transplantation & Conversion Training
```bash
tmux new -s train
bash scripts/linux/transplant_train.sh 2>&1 | tee outputs/logs/train.log
```

### Step 4: Validation and Export
```bash
bash scripts/linux/validate_export.sh 2>&1 | tee outputs/logs/validate_export.log
```

---

## 6. Running the Pipeline (Windows/PowerShell)

For local development or testing on a Windows machine:

### Step 1: Environment Setup
```powershell
.\scripts\windows\setup.ps1
```

### Step 2: Teacher Generation (via local LM-Studio server)
1. Open LM-Studio and start the local server on port `1234`.
2. Load the model `google/gemma-4-e4b`.
3. Run the generation script:
```powershell
.\scripts\windows\generate.ps1
```

### Step 3: Block Corruption
```powershell
.\scripts\windows\corrupt.ps1
```

### Step 4: Weight Transplantation & LoRA Training
```powershell
.\scripts\windows\transplant_train.ps1
```

### Step 5: Validation and Export
```powershell
.\scripts\windows\validate_export.ps1
```

---

## 7. Monitoring Training Progress

Run the monitor script on your Linux machine to watch GPU utilization, disk usage, and loss metrics:

```bash
bash scripts/linux/monitor.sh
```

---

## 8. Failure Recovery & Troubleshooting

### Out of Memory (OOM) on 48GB GPU
If you encounter OOM during QLoRA training on 48GB GPUs, increase gradient accumulation steps:
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

### Server Disk Space Cleanup
If the disk space runs low, compress and clean up old checkpoints:
```bash
df -h .
tar -czf old-checkpoints.tar.gz artifacts/conversion_training/checkpoint-0000*
rm -rf artifacts/conversion_training/checkpoint-0000*
```
