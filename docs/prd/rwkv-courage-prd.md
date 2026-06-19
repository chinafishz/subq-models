# RWKV-Courage 需求文檔

## 1. 背景

### 1.1 當前問題
現有語言模型多為 Transformer 架構，注意力複雜度 O(n²)，訓練成本高昂。SubQ-1.1-Small 論證了次平方稀疏注意力（SSA）可實現線性計算 + 內容相關路由 + 長度泛化。本專案探索以 RWKV-7「Goose」架構（首個端到端 O(n) 注意力、meta-in-context learner）從零訓練一個具備「勇氣」人格的微型語言模型。

### 1.2 影響範圍
- 全新建構，不影響任何現有系統
- 目標硬體：Mac Mini M2 16GB 統一記憶體
- 目標平台：純 PyTorch + MPS 後端

## 2. 目標

### 2.1 本次要解決什麼

| ID | 目標 |
|----|------|
| REQ-001 | 從零預訓練 RWKV-7 ~25M 參數模型，在 TinyStories 上達到可用 perplexity |
| REQ-002 | 注入基於數碼暴龍八徽章哲學的「勇氣」人格（15% 訓練語料混合比） |
| REQ-003 | 模型能產出展現勇氣特質的連貫文本（敢嘗試、敢行動、失敗不放棄） |
| REQ-004 | 全流程在 Mac Mini M2 16GB 上可執行（純 PyTorch MPS） |
| REQ-005 | 自訂 BPE 8K 分詞器，與模型 embedding 層對齊 |

### 2.2 本次不解決什麼

| Non-Goal | 說明 |
|----------|------|
| Non-Goal-001 | 暫不接入 FineWeb-Edu（第一階段僅 TinyStories 驗證管線） |
| Non-Goal-002 | 暫不做 instruction tuning / RLHF（純預訓練階段） |
| Non-Goal-003 | 不追求 SOTA benchmark 分數（目標是人格驗證） |
| Non-Goal-004 | 不做多語言（純英文） |

## 3. 業務規則

| ID | 規則 |
|----|------|
| BR-001 | 模型參數約 25M（n_embd=384, n_layer=8, n_head=6） |
| BR-002 | 上下文長度 4K tokens |
| BR-003 | 訓練資料比例：85% TinyStories + 15% Digimon 勇氣材料 |
| BR-004 | 分詞器：自訂 BPE，vocab_size=8000，在混合語料上訓練 |
| BR-005 | 訓練在純 PyTorch MPS 上執行（不依賴 CUDA） |
| BR-006 | 損失函數：標準 cross-entropy（sample-level aggregation） |
| BR-007 | 優化器：AdamW，學習率 3e-4，warmup 500 steps，cosine decay |

## 4. 驗收標準（AC）

### AC-001：分詞器訓練與驗證
**Given** 混合訓練語料（TinyStories + Digimon 材料）  
**When** 執行 `train_tokenizer.py`  
**Then** 產出 `tokenizer.json`，vocab_size=8000，encode/decode 往返一致  

### AC-002：RWKV-7 模型初始化
**Given** 模型配置（n_embd=384, n_layer=8, n_head=6, vocab_size=8000）  
**When** 調用 `CourageLM(config)`  
**Then** 參數量 ≈ 25M，forward pass 無報錯，輸出 shape = (batch, seq_len, vocab_size)  

### AC-003：TinyStories 訓練管線
**Given** TinyStories 訓練資料 + 自訂 BPE tokenizer  
**When** 執行 `train.py` 訓練 1 epoch  
**Then** 損失下降，無 NaN，訓練速度穩定  

### AC-004：勇氣人格注入
**Given** 訓練資料含 15% Digimon 勇氣材料  
**When** 模型預訓練完成後，輸入 prompt "I believe that"  
**Then** 續寫文本展現勇氣/探索/不放棄相關語義  

### AC-005：Mac M2 MPS 兼容
**Given** Mac Mini M2 16GB，torch.backends.mps.is_available() = True  
**When** 執行完整訓練管線  
**Then** 不報 CUDA 相關錯誤，訓練正常完成  

### AC-006：Checkpoint 儲存與恢復
**Given** 訓練中斷  
**When** 從 checkpoint 恢復訓練  
**Then** 損失和優化器狀態正確恢復，訓練可繼續  

## 5. 測試矩陣

| TC | 對應 AC | 類型 | 輸入/狀態 | 預期結果 | 狀態 |
|----|---------|------|-----------|---------|------|
| TC-001 | AC-001 | 單元 | 混合語料 10MB | 產出 tokenizer.json，vocab=8000 | 未開始 |
| TC-002 | AC-002 | 單元 | config dict | 模型參數 25M ± 2M | 未開始 |
| TC-003 | AC-003 | 整合 | TinyStories 子集 10K stories | loss 從 ~8 降至 ~4 | 未開始 |
| TC-004 | AC-004 | 整合 | prompt="I believe" | 文本含 courage/believe/try/never give up | 未開始 |
| TC-005 | AC-005 | 環境 | `torch.backends.mps.is_available()` | return True | 未開始 |
| TC-006 | AC-006 | 整合 | 訓練到 step 500 → 中斷 → 恢復 | loss 連續 | 未開始 |

## 6. 實現方案

### 6.1 專案結構
```
rwkv-courage/
├── src/
│   ├── model.py          # RWKV-7 模型定義（純 PyTorch）
│   ├── wkv7_operator.py  # RWKV-7 WKV/Delta Rule 算子
│   ├── tokenizer_train.py # BPE 分詞器訓練
│   ├── dataset.py        # 資料載入與混合
│   └── train.py          # 訓練迴圈
├── data/
│   ├── raw/              # 原始語料
│   └── tokenized/        # .bin/.idx 格式
├── configs/
│   └── courage_25m.yaml  # 模型 + 訓練配置
├── scripts/
│   └── prepare_data.sh   # 資料下載與預處理
├── checkpoints/          # 模型保存
└── requirements.txt
```

### 6.2 模組依賴
```
tokenizer_train.py → dataset.py → train.py
model.py ← wkv7_operator.py → train.py
```

### 6.3 技術選型
- 分詞器：HuggingFace `tokenizers` (BPE, vocab_size=8000)
- RWKV-7 核心：基於官方 `rwkv_v7_demo.py` 改寫純 PyTorch 版本
- 訓練框架：純 PyTorch train loop（不引入 Lightning，減少依賴）
- 日誌：WandB（可選）/ CSV logging
- 資料格式：二進制 `.bin/.idx`（RWKV 原生格式）

## 7. Loop 記錄

| Loop | 內容 | AC 狀態 |
|------|------|---------|
| - | 初始 PRD 建立 | - |

## 8. 風險與邊界情況

| Risk | 描述 | 緩解 |
|------|------|------|
| Risk-001 | RWKV-7 WKV 算子無純 PyTorch 實現 | 以官方 `rwkv_v7_demo.py` 的非 CUDA 路徑為基礎改寫；25M 模型 for 迴圈可接受 |
| Risk-002 | MPS 後端數值精度問題（BF16 不支援） | 使用 FP32 訓練；25M 模型記憶體充足 |
| Risk-003 | 8K 詞表覆蓋不足 | 監控 UNK token 比例；若 > 2% 則提高至 12K |
| Risk-004 | 人格材料 15% 比例不足夠錨定人格 | 訓練後用 prompt 探針評估；若不夠則後續 FineWeb-Edu 階段提高至 25% |
| Risk-005 | 自訂 BPE tokenizer 與 RWKV binidx 格式不相容 | 先驗證 tokenize → detokenize 往返一致性再轉 binidx |
| Risk-006 | 16GB 不足以同時處理 FineWeb-Edu（後續階段） | 第一階段僅 TinyStories；FineWeb-Edu 階段串流處理 |

## 9. 最終驗收清單

- [ ] AC-001 至 AC-006 全部通過
- [ ] 模型可在 Mac M2 上完整訓練並推理
- [ ] 勇氣人格可透過 prompt 探針檢測
- [ ] 所有程式碼和文檔推送至 chinafishz/subq-models
