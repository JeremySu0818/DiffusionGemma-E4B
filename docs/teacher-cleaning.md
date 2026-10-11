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

清理命令不修改 progress 或 resume 索引。若之後再續跑 teacher，替換原檔造成的檔案 identity 變更會讓既有 resume 索引從 JSONL 重新建構，已刪除的 prompt 可以重新生成。
