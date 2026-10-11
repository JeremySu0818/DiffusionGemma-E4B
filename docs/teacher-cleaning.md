# Teacher 生成後清理

生成程序結束後執行；不要對正在追加寫入的 JSONL 執行。工具直接更新指定原檔，不保留備份或隔離資料集。逐行過濾寫入同目錄暫存檔，完成後替換原檔，暫存檔不會保留。

```bash
.venv/bin/python -m diffusiongemma_e4b.clean_teacher \
  --jsonl data/teacher_supervised/large/teacher_outputs.jsonl \
  --tokenizer-json artifacts/tokenizer_processor_gemma4_e4b/tokenizer.json
```

必須使用與生成時 `teacher_input_tokens` 計數一致的 tokenizer。工具重建當時的完整題目；完整題目 tokens 大於記錄的實際輸入 tokens 時，刪除該筆輸入截短的資料。舊版相同 context/prompt 仍有包裝文字，新版依 `teacher_input_policy` 重建。

預設保留 `teacher_finish_reason=length` 的部分回答：它不代表回答內容錯誤，新的 corruption 已不替它加上結束標記。若只要完整回答，在上述命令加上 `--drop-length-limited`，直接刪除這些記錄。

沒有輸入計數、沒有 finish reason，或計數方向相反的記錄只列入終端報告，不猜測或自動刪除。沒有長度、重複率、配比、短答案等品質門檻。保留記錄的 JSONL 內容不改寫。

此工具不能從舊 JSONL 判斷已消失的來源內容、被壓平的換行或答案正確性；這些需要與上游原始題目比較或重新生成。

清理後，用新目錄製作完整輸入的訓練 shards：

```bash
DG_TEACHER_OUTPUT=data/teacher_supervised/large/teacher_outputs.jsonl \
DG_CORRUPTION_DIR=data/corruption/large_clean \
DG_TARGET_BLOCKS=0 bash scripts/linux/corrupt.sh
```

全量製作先掃描 JSONL 計算總筆數，進度條顯示筆數、百分比、速度、ETA 和累計 blocks。ETA 估算資料處理階段；最後寫檔及驗證另有階段提示。

新版 shards 使用 schema 5，只保存乾淨的 prompt、target、媒體索引、記錄資訊和 tokenizer 詞表大小，不產生或保存 `corrupted_ids`、`corruption_masks`、`noise_t`。圖片／音訊及其位置／特徵遮罩張量使用內容 SHA-256 命名，壓縮存於輸出目錄的 `media/`；相同張量跨 block、record、shard 只存一份。隨 prompt 長度改變的 `mm_token_type_ids` 仍保存於各 block。搬移訓練資料時必須連同 `media/` 一起搬移；驗證會檢查引用檔案與內容雜湊，訓練指紋也涵蓋媒體檔案。此去重不改變浮點張量數值。

訓練及 forward 驗證時才即時加噪。離線噪聲選項與讀取分支已移除；訓練依 seed、epoch、shard 和 row 確定噪聲，同一 epoch 可重現，下一 epoch 重新取樣。schema 與資料指紋已更新，重新執行相同製作命令會建立新版 shards，不續接 schema 4 或舊版帶噪資料的 checkpoint；舊暫存會保留在原目錄。讀取端仍支援 schema 4 的乾淨內嵌媒體 shards。

寫入前會檢查磁碟可用空間，並計入同時寫檔的保守預算；預設保留 2 GiB，空間不足時停止並保留最近成功的 checkpoint。釋放空間後可續跑。Linux/Windows 使用 `DG_MIN_FREE_DISK_BYTES`，CLI 使用 `--min-free-disk-bytes`；此限制可在續跑時調整。`DG_MAX_MULTIMODAL_SHARD_BYTES`／`--max-multimodal-bytes` 是已棄用的相容參數，並非磁碟容量保護；`multimodal_uncompressed_bytes` 統計的是各 block 的邏輯張量大小，包含共用張量的重複引用，不等於實際檔案用量。

Linux/Windows 腳本預設使用全部可用 CPU：多程序執行 tokenizer 與媒體預處理，多執行緒壓縮、寫入及驗證 shards。Linux worker 繼承載入好的 tokenizer/processor，避免逐 worker 複製大型詞表。主程序依輸入順序收取結果，保持資料與續跑一致。預取及尚未完成的寫檔數量都有界限。

`DG_CORRUPTION_WORKERS=0` 自動選擇可用 CPU 數，`1` 改為串行，正值指定 worker 數；CLI 對應 `--num-workers`。`DG_CORRUPTION_PREFETCH=0` 預設預取同 worker 數的記錄，正值指定數量；CLI 對應 `--prefetch-records`。降低預取數可以降低處理中媒體佔用的 RAM。worker 數與預取數不影響資料指紋，可在續跑時調整。

```bash
DG_CORRUPTION_WORKERS=0 \
DG_TEACHER_OUTPUT=data/teacher_supervised/large/teacher_outputs.jsonl \
DG_CORRUPTION_DIR=data/corruption-large \
DG_TARGET_BLOCKS=0 bash scripts/linux/corrupt.sh
```

腳本預設把每個 worker 內的 OpenMP/MKL/OpenBLAS threads 設為 1，避免多層並行互相搶 CPU。計數、輸入 SHA-256 與最終發布仍由主程序處理；磁碟 I/O、記憶體頻寬及記錄大小也會影響 CPU 利用率。

Shards 現在會自動續跑，使用完全相同的命令即可。每約 60 秒在完整記錄邊界將 buffers 寫成 shards，並原子保存 `corruption_checkpoint.json`，包括已讀筆數、shard 清單及統計。首次 Ctrl+C 會等待當前記錄完成並保存後退出；再次 Ctrl+C 或強制終止，則下次從最後成功的 checkpoint 恢復，尚未保存的工作會重做。

恢復前會重新確認輸入及參數指紋、已保存 shard 的大小和 SHA-256。已完成的記錄會跳過，中斷後尚未提交的 shard 會重新製作。改變資料或參數會使用另一個暫存目錄。舊版只保存 `corruption_progress.json` 的暫存無法恢復，程式會保留在 `.legacy-*` 目錄並重新開始。

清理命令不修改 progress 或 resume 索引。若之後再續跑 teacher，替換原檔造成的檔案 identity 變更會讓既有 resume 索引從 JSONL 重新建構，已刪除的 prompt 可以重新生成。
