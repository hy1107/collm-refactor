# CoLLM Refactor

本專案是 [CoLLM](https://github.com/zyang1580/CoLLM)（Collaborative Large Language Model for Recommendation）的重構版本，使用 Python 3.10，以 HuggingFace Trainer 取代原始自定義 Runner，採用模塊化四層架構。

---

## 架構概覽

```
collm/
├── collm/
│   ├── constants.py            # 全域常數（USER_TOKEN, ITEM_TOKEN）
│   ├── encoders/               # 推薦 Encoder 層
│   │   ├── base.py             # BaseRecEncoder 抽象類
│   │   ├── mf.py               # Matrix Factorization
│   │   ├── lightgcn.py         # LightGCN（圖卷積）
│   │   ├── sasrec.py           # SASRec（自注意力序列）
│   │   └── din.py              # DIN（深度興趣網絡）
│   ├── model/
│   │   ├── cie.py              # CIE 投影模塊（d_rec → d_llm）
│   │   ├── backbone.py         # LLM 載入工廠（LoRA + 特殊 token）
│   │   └── collm.py            # CoLLMModel 主模型
│   ├── data/
│   │   ├── prompts.py          # Prompt 模板（含/不含占位符）
│   │   ├── dataset.py          # RecDataset（MovieLens / Amazon）
│   │   └── collator.py         # RecDataCollator
│   └── training/
│       ├── config.py           # CoLLMConfig dataclass（YAML 載入）
│       ├── metrics.py          # AUC, HR@K, NDCG@K
│       └── trainer.py          # CoLLMTrainer（繼承 HF Trainer）
├── scripts/
│   ├── train_rec_encoder.py    # Rec Encoder 預訓練
│   ├── train_stage1.py         # Stage 1：LoRA 預訓練（純文字）
│   └── train_stage2.py         # Stage 2：CIE 訓練（帶協同信號）
├── configs/
│   ├── stage1_movielens.yaml
│   ├── stage2_movielens.yaml
│   └── rec_encoder_movielens.yaml
└── tests/                      # 43 個單元測試
```

---

## 環境安裝

```bash
# 建議使用虛擬環境
python3.10 -m venv venv
source venv/bin/activate

# 安裝套件（含開發工具）
pip install -e ".[dev]"
```

**主要依賴：** Python 3.10、PyTorch 2.0+、transformers 4.35+、peft 0.6+、dacite 1.8+

---

## 訓練流程

CoLLM 訓練分三個階段：

### 階段一：預訓練 Rec Encoder

使用 BPR loss 在交互數據上訓練推薦 Encoder（MF 或 SASRec）：

```bash
python scripts/train_rec_encoder.py \
    --data_path /data/ml-1m/train \
    --encoder_type sasrec \
    --embedding_dim 64 \
    --n_layers 2 \
    --n_heads 2 \
    --max_seq_len 50 \
    --epochs 50 \
    --lr 1e-3 \
    --save_path /checkpoints/sasrec_ml1m.pth
```

支援的 encoder 類型：`mf`、`sasrec`

### 階段二：Stage 1 — LoRA 預訓練（純文字）

凍結 LLM，僅用純文字 prompt 微調 LoRA，不使用協同信號：

```bash
python scripts/train_stage1.py \
    --config configs/stage1_movielens.yaml \
    --output_dir /checkpoints/stage1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4
```

### 階段三：Stage 2 — CIE 訓練（帶協同信號）

載入 Stage 1 checkpoint，凍結 backbone + rec encoder，只訓練 CIE 投影層：

```bash
python scripts/train_stage2.py \
    --config configs/stage2_movielens.yaml \
    --stage1_checkpoint /checkpoints/stage1 \
    --output_dir /checkpoints/stage2 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4
```

---

## YAML 配置

所有超參數集中在 YAML 配置檔，透過 dacite 載入為 dataclass：

```yaml
# configs/stage2_movielens.yaml
rec:
  encoder_type: sasrec
  checkpoint_path: /checkpoints/sasrec_ml1m.pth
  embedding_dim: 64
  use_collaborative_signal: true   # false = 消融實驗模式
  user_num: 0                      # 執行時自動填入
  item_num: 0
  max_seq_len: 50
  n_layers: 2
  n_heads: 2
  dropout: 0.2
  n_gcn_layers: 3

backbone:
  model_name_or_path: lmsys/vicuna-7b-v1.3
  lora_r: 8
  lora_alpha: 16
  lora_target_modules:
    - q_proj
    - v_proj
  load_in_8bit: false

data:
  data_path: /data/ml-1m
  dataset_type: movielens
  max_history_len: 20
  max_seq_len: 512
```

在 Python 中載入：

```python
import yaml, dacite
from collm.training.config import CoLLMConfig

raw = yaml.safe_load(open("configs/stage2_movielens.yaml"))
cfg = dacite.from_dict(CoLLMConfig, raw, config=dacite.Config(cast=[list]))
```

---

## 消融實驗

切換 `use_collaborative_signal: false` 即可移除協同信號，自動套用純文字 prompt 模板並跳過 embedding 替換：

```yaml
# 在 YAML 中修改
rec:
  use_collaborative_signal: false
```

或在訓練腳本中覆蓋：

```python
cfg.rec.use_collaborative_signal = False
```

---

## 新增 Rec Encoder

1. 在 `collm/encoders/` 建立新文件（如 `dcn.py`）
2. 繼承 `BaseRecEncoder`，實作三個 abstractmethod：
   - `get_user_embedding(user_ids, **kwargs) → Tensor`
   - `get_item_embedding(item_ids, **kwargs) → Tensor`
   - `embedding_dim` property
3. 在 `collm/encoders/__init__.py` 匯出
4. 在 `scripts/train_stage2.py` 的 `_ENCODER_MAP` 加入對應項目
5. 在 `tests/test_encoders.py` 補充測試

---

## 測試

```bash
python3.10 -m pytest tests/ -v
# 43 passed
```

---

## 原始論文

> **CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation**
> Yang et al., 2023 — [arXiv](https://arxiv.org/abs/2310.19488) | [原始代碼](https://github.com/zyang1580/CoLLM)

本重構保留論文核心架構（兩階段訓練、CIE 投影、占位符注入），以更清晰的模塊邊界和現代訓練框架重新實現。
