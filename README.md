# DiffusionGemma E4B

這個專案把 `google/gemma-4-E4B-it` 的權重與行為轉成一個獨立、E4B
規模的多模態 Diffusion Transformer。它不是 26B/A4B 模型，也不是把
Gemma 4 當成一般 causal LM 繼續微調。

Linux 雲端 GPU 的正式流程維持一條主指令：

```bash
bash scripts/linux/run_pipeline.sh
```

流程會依序執行 preflight、啟動或連接 Gemma 4 E4B teacher、產生蒸餾
資料、建立 diffusion corruption shards、移植 E4B 權重、QLoRA 訓練、
strict-diffusion 推論驗證，以及包含 base model 的原子化 release 匯出。
任一必要契約不成立就停止，不會把不完整 checkpoint 包裝成 final。

## 實際訓練的模型

正式 student 為本專案的
`MultimodalDiffusionGemmaForBlockDiffusion`：

- 權重來源與資料 teacher 都是 `google/gemma-4-E4B-it`。
- 保留 E4B 的 42 層、hidden size 2560、intermediate size 10240、
  dense MLP、Per-Layer Embeddings（PLE）及 18 個 shared-KV layers。
- 保留 Gemma 4 E4B 的 vision tower、audio tower 與兩者的 projector。
- prompt encoder 保持原本的 causal attention。
- diffusion decoder 讀取 encoder KV cache，256-token canvas 內使用
  bidirectional attention，不修改 encoder cache。
- encoder 與 decoder 的 E4B-compatible base weights 綁定，不會因兩條
  路徑變成兩倍參數；新增的主要參數是 diffusion self-conditioning。
- 訓練只在 diffusion decoder 的 linear layers 加 LoRA/QLoRA adapter；
  LM head 與多模態 towers 不會被意外列入 LoRA target。

移植必須得到所有 target tensors 的完整覆蓋、零 missing source、零 shape
mismatch。正式 pipeline 不接受 partial transplant。

## 雲端需求

- Linux、Python 3.11 或 3.12。
- NVIDIA CUDA GPU 且原生支援 BF16。
- 預設 preflight 要求總可見 VRAM 至少 75 GiB、啟動前 free VRAM 至少
  70 GiB；建議 A100/H100 80GB。多 GPU teacher 的 tensor parallel
  必須與可見 GPU 數相容。
- host RAM 至少 64 GiB，啟動前 available RAM 至少 32 GiB。移植階段使用
  meta-device 建模，避免綁定權重前短暫配置兩份 E4B student。
- `gpu` preset 建議至少 200 GiB 可用磁碟；preflight 會分別檢查
  Hugging Face cache、media、corruption、training 與 export 所在 mount。
- 已接受 `google/gemma-4-E4B-it` 的 Hugging Face gated model 條款並執行
  `hf auth login`。
- 已取得所有啟用資料來源的存取權；preflight 會即時驗證 config 與 split。

多 GPU 本機 teacher 例子：

```bash
export DG_TEACHER_TENSOR_PARALLEL_SIZE=2
bash scripts/linux/run_pipeline.sh
```

## 安裝與執行

全新的 GPU instance：

```bash
hf auth login
bash scripts/linux/setup.sh
bash scripts/linux/run_pipeline.sh
```

`setup.sh` 會建立 `.venv`，以同一個 resolver transaction 安裝 PyTorch、
Transformers、vLLM、PEFT 與 bitsandbytes，最後執行 dependency check。
若 image 已備妥 OS 套件，可設定 `DG_SKIP_APT=1`。

先跑最小端到端接線測試：

```bash
DG_PRESET=smoke bash scripts/linux/run_pipeline.sh
```

Smoke 仍會載入與移植真正的 Gemma 4 E4B，所以需要 GPU；它只縮小資料量
和 optimizer updates。Smoke、`gpu`、`large` 的 teacher、corruption、
training、validation 與 export 路徑彼此隔離。

### Presets

| Preset | Teacher tokens | Canvas blocks | Updates | QLoRA | 用途 |
|---|---:|---:|---:|---:|---|
| `gpu`（預設） | 60M | 220k × 256 | 13k，accum 16 | r64 / α128，LR 1e-4 | 約 0.95 個 block pass |
| `large` | 550M | 2M × 256 | 125k，accum 32 | r64 / α128，LR 5e-5 | 約 2 個 block passes |
| `smoke` | 32,768 | 128 × 256 | 2，accum 1 | 接線測試 | 驗證真實模型完整流程 |

若目標是最高教師行為覆蓋而成本允許：

```bash
DG_PRESET=large bash scripts/linux/run_pipeline.sh
```

不要啟用 `DG_GRADIENT_CHECKPOINTING=1`。通用 Transformers checkpoint
wrapper 會丟掉 diffusion decoder 必須讀取的 encoder KV cache；preflight
會直接拒絕這個設定。

## Teacher endpoint

預設由 vLLM 在本機啟動 Gemma 4 E4B。資料生成完成後，pipeline 只會停止
自己啟動的 process，再釋放 GPU 給 student。使用外部 OpenAI-compatible
endpoint：

```bash
export DG_SKIP_LOCAL_TEACHER=1
bash scripts/linux/setup.sh

export DG_TEACHER_BASE_URL=https://teacher.example/v1
export DG_TEACHER_SERVED_MODEL_NAME=gemma4-e4b
export DG_TEACHER_API_KEY=...
bash scripts/linux/run_pipeline.sh
```

API key 不會寫入輸出。Readiness probe 會檢查 `/models` 確實包含指定
model ID。預設並行 8 個請求，每個 worker 有獨立 HTTP session；每筆成功
輸出都會 durable append 與 `fsync`。重啟以 prompt ID、資料與 generation
fingerprint 接續。送給 teacher 的 prompt 會先依 student prefix budget
截斷，避免 teacher 使用 student 看不到的 context。

## 資料配置

[`configs/dataset_sources.json`](configs/dataset_sources.json) 的 prompt
record 配比為：

- 69% 一般對話、instruction following、知識與 long-context。
- 30% reasoning、數學、科學與程式碼。
- 1% image、文件 OCR、chart 與 diagram understanding。

第三方 assistant answer 不會當訓練 target；上游資料只供 prompt、context
和 media，target 一律由 Gemma 4 E4B teacher 重新產生。資料準備器以 bucket
quota 交錯取樣；teacher progress 記錄成功筆數與 token 數；corruption
manifest 再記錄實際 training-block 比例。必要 bucket 消失、嚴重低於配比，
或無法精確建立指定 block 數時會停止。

E4B audio encoder 權重有保留，但 audio source 預設關閉。只有在選定
Common Voice language config/split，且確認 teacher endpoint 能接收 audio
後才應另開實驗；預設正式資料不會假裝具有尚未驗證的 audio 蒸餾。

請依用途確認每個上游資料集當下的 license、地域、存取與商用條件。
`license_hint` 是操作提醒，不是法律判定。

## 訓練與 release gates

預設訓練包含：

- NF4 double-quantized QLoRA。
- online uniform-state corruption，noise `t ∈ [0.05, 0.95]`。
- 只對被 corruption 的有效 target token 計算 normalized denoising CE。
- 50% self-conditioning；第一次 forward 在 `no_grad` 下執行。
- 以 source record ID 切 train/validation，避免同一回答的相鄰 blocks
  同時出現在兩側。
- 固定 seed、按 bucket 分層的 128 個 held-out blocks 作 checkpoint
  selection；正式 release 要求相對 baseline 至少改善 0.5%。
- optimizer、scheduler、AMP scaler、Python/NumPy/Torch/CUDA RNG 與資料
  cursor 的 exact resume。

Step 0 的零初始化 LoRA 是 E4B transplant baseline。只有 best held-out
denoising loss 相對 baseline 達到設定門檻才建立 `final`；Smoke 門檻為 0，
只驗證接線。Release 還必須通過：

- corruption manifest 的 byte size、SHA-256、array shape 與精確 block count。
- model、processor/chat template、資料與訓練 fingerprint 一致性。
- E4B 拓撲、audio tower、自訂 decoder 及 encoder/decoder tied weights。
- finite masked-denoising forward。
- `DiffusionGemmaGenerationConfig` 與 entropy-bound sampler 契約。
- 48-step strict diffusion 產生非空文字，且沒有 AR fallback。
- release archive readback 與逐檔 SHA-256。

## Resume、輸出與使用

中斷後使用同一條指令：

```bash
bash scripts/linux/run_pipeline.sh
```

主要輸出：

- `data/teacher_supervised/teacher_outputs.jsonl`：durable teacher targets。
- `data/teacher_supervised/progress.json`：generation 與 bucket 進度。
- `data/corruption/`：已驗證 NPZ shards 與 manifest。
- `artifacts/transplanted/`：完整 E4B Diffusion Transformer base。
- `artifacts/conversion_training/checkpoint-*`：可精確恢復的 checkpoint。
- `artifacts/conversion_training/best/`：最低 held-out loss adapter。
- `artifacts/conversion_training/final/`：通過改善門檻的 final adapter。
- `outputs/validation/`：架構、forward、generation 與 strict inference 報告。
- `artifacts/diffusiongemma-e4b-repro-bundle.tar.gz`：包含 final adapter、
  `artifacts/base_model/`、程式碼、設定與驗證報告的可攜 bundle。

Final 預設是 PEFT adapter；單獨在原工作目錄載入時，它的 base 是
`artifacts/transplanted`。解開 release bundle 後，loader 會自動找到
`artifacts/base_model`；也可以明確指定：

```bash
python -m diffusiongemma_e4b.infer \
  --model-dir artifacts/final \
  --base-model artifacts/base_model \
  --load-in-4bit \
  --prompt "Explain block diffusion briefly."
```

若要改資料、模型、tokenizer 或會影響訓練的超參數，應使用新的
`DG_TEACHER_OUTPUT`、`DG_TEACHER_PROGRESS`、`DG_CORRUPTION_DIR` 與
`DG_TRAIN_OUTPUT_DIR`。Fingerprint 不一致時流程會拒絕混用舊進度。

## 可誠實保證的範圍

Repository 內的靜態契約、tiny E4B forward/save/reload/strict-generation、
資料 durability、resume 與 release gates 可在 CPU 測試。完整 E4B
權重移植也會在執行時逐 tensor 驗證。

沒有實際跑完目標雲端 GPU、完整 teacher generation 與訓練前，不能誠實
宣稱零 runtime error 或品質已等同 Gemma 4 E4B。Pipeline 的做法是讓未知
問題 fail closed，並把「可發布」最低定義成：E4B 架構與權重契約成立、
held-out teacher denoising loss 相對 baseline 改善、strict diffusion
生成通過。若要對外宣稱任務品質接近 teacher，仍應在完成後另跑
task-level benchmark、人工偏好與安全評測。
