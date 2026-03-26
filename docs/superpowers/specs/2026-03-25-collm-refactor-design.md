# CoLLM 重構設計文件

**日期**：2026-03-25
**狀態**：已確認（v2，修正審核問題後）
**語言**：Python 3.10

---

## 一、背景與目標

CoLLM（Collaborative Large Language Model for Recommendation）原始代碼繼承自 MiniGPT-4 / Lavis 框架，存在以下問題：

- 大量視覺模型殘留代碼（ViT、Q-Former、圖片處理等）與推薦任務無關
- 自定義 Registry 注冊系統讓調用鏈難以追蹤
- 四個 Rec Encoder 混在同一文件，輸出格式不統一
- 自定義 Runner 缺乏現代訓練功能
- OmegaConf 配置缺少類型提示與 IDE 支持

**重構目標**：

1. **模塊化**：四層清晰分離（encoders / model / data / training），每層只做一件事，透過明確介面通信
2. **研究友好**：使用 HuggingFace Trainer，天然支持 LoRA/PEFT、混合精度、分散式訓練、checkpoint 管理
3. **可擴展**：LLM backbone 可配置（Vicuna / Qwen / 其他 HF 模型），Rec Encoder 統一介面方便新增模型
4. **消融實驗支持**：單一參數切換「有無協同 embedding」

---

## 二、目錄結構

```
collm/
├── collm/
│   ├── encoders/               # Rec Encoder 層
│   │   ├── __init__.py
│   │   ├── base.py             # BaseRecEncoder 抽象類
│   │   ├── mf.py               # MatrixFactorization
│   │   ├── lightgcn.py         # LightGCN
│   │   ├── sasrec.py           # SASRec
│   │   └── din.py              # DIN
│   │
│   ├── model/                  # 核心模型層
│   │   ├── __init__.py
│   │   ├── cie.py              # CIE 投影模塊
│   │   ├── collm.py            # 主模型：組裝 encoder + CIE + LLM backbone
│   │   └── backbone.py         # LLM backbone 工廠函數（含特殊 token 注冊）
│   │
│   ├── data/                   # 數據層
│   │   ├── __init__.py
│   │   ├── dataset.py          # MovieLens / Amazon Dataset 類
│   │   ├── collator.py         # DataCollator（prompt 組裝、占位符位置）
│   │   └── prompts.py          # Prompt 模板管理（含/不含占位符兩種版本）
│   │
│   ├── training/               # 訓練層
│   │   ├── __init__.py
│   │   ├── trainer.py          # 繼承 HF Trainer，覆蓋 compute_loss
│   │   ├── config.py           # 所有 dataclass 配置定義
│   │   └── metrics.py          # AUC、HR、NDCG 評估指標
│   │
│   └── __init__.py
│
├── scripts/
│   ├── train_rec_encoder.py    # 預訓練 Rec Encoder（BPR/BCE loss，獨立於 CoLLM）
│   ├── train_stage1.py         # Stage 1 入口（LoRA 預訓練，純文字）
│   └── train_stage2.py         # Stage 2 入口（CIE 訓練，帶協同信號）
│
├── configs/                    # YAML 範例配置
│   ├── rec_encoder_movielens.yaml
│   ├── stage1_movielens.yaml
│   └── stage2_movielens.yaml
│
├── tests/
│   ├── test_encoders.py
│   ├── test_cie.py
│   └── test_collm_model.py
│
└── pyproject.toml
```

---

## 三、核心介面設計

### 3.1 BaseRecEncoder（`encoders/base.py`）

所有 Rec Encoder 的抽象基類，統一輸出格式：

```python
class BaseRecEncoder(nn.Module, ABC):
    @abstractmethod
    def get_user_embedding(self, user_ids: Tensor) -> Tensor:
        """輸入: (batch,) → 輸出: (batch, d_rec)"""

    @abstractmethod
    def get_item_embedding(self, item_ids: Tensor, **kwargs) -> Tensor:
        """輸入: (batch,) → 輸出: (batch, d_rec)
        SASRec/DIN 等序列模型透過 kwargs 傳入 seq_history"""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """返回 d_rec"""
```

各模型實現：
- `MFEncoder`（mf.py）：標準矩陣分解
- `LightGCNEncoder`（lightgcn.py）：圖卷積，需傳入鄰接矩陣
- `SASRecEncoder`（sasrec.py）：自注意力序列模型，`get_item_embedding` 接受 `seq_history` kwarg
- `DINEncoder`（din.py）：深度興趣網絡，`get_item_embedding` 接受 `seq_history` kwarg

### 3.2 CIEModule（`model/cie.py`）

將協同 embedding 投影到 LLM token 空間：

```python
class CIEModule(nn.Module):
    def __init__(self, rec_dim: int, llm_dim: int):
        self.proj = nn.Linear(rec_dim, llm_dim)

    def forward(self, rec_embedding: Tensor) -> Tensor:
        """(batch, d_rec) → (batch, d_llm)"""
        return self.proj(rec_embedding)
```

> **注意**：原始論文使用 Q-Former 風格的多層投影橋接。此處簡化為單一線性層，實現更清晰、參數更少，但可能與論文原始數字略有差異。若需完全復現原始結果，可將此模塊替換為多層 MLP 或 Q-Former，介面保持不變。

### 3.3 CoLLMModel（`model/collm.py`）

主模型，組裝所有模塊。`rec_encoder` 和 `cie_module` 均為 Optional，支持 Stage 1 純文字模式：

```python
class CoLLMModel(nn.Module):
    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        rec_encoder: Optional[BaseRecEncoder] = None,   # Stage 1 時為 None
        cie_module: Optional[CIEModule] = None,          # Stage 1 時為 None
        config: Optional[CoLLMConfig] = None,
    ): ...

    def forward(
        self,
        input_ids: Tensor,                        # (batch, seq_len)
        attention_mask: Tensor,                   # (batch, seq_len)
        labels: Tensor,                           # (batch, seq_len)
        user_ids: Optional[Tensor] = None,        # (batch,)
        target_item_ids: Optional[Tensor] = None, # (batch,)
        user_placeholder_pos: Optional[Tensor] = None,  # (batch,) LongTensor，-1 表示無占位符
        item_placeholder_pos: Optional[Tensor] = None,  # (batch,) LongTensor，-1 表示無占位符
        seq_history: Optional[Tensor] = None,     # (batch, max_history)，僅 SASRec/DIN 使用
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # 1. 取 LLM token embeddings: embeds = backbone.get_input_embeddings()(input_ids)
        # 2. 若 rec_encoder 不為 None 且 self.use_collaborative_signal=True：
        #    （self.use_collaborative_signal 在 __init__ 中從 config.rec.use_collaborative_signal 讀取並存為實例變數）
        #    a. user_emb = rec_encoder.get_user_embedding(user_ids)           → (batch, d_rec)
        #    b. item_emb = rec_encoder.get_item_embedding(target_item_ids, seq_history=seq_history) → (batch, d_rec)
        #    c. user_token = cie_module(user_emb)                             → (batch, d_llm)
        #    d. item_token = cie_module(item_emb)                             → (batch, d_llm)
        #    e. 對 user_placeholder_pos != -1 的樣本，替換對應位置的 embedding
        #    f. 對 item_placeholder_pos != -1 的樣本，替換對應位置的 embedding
        # 3. backbone.forward(inputs_embeds=embeds, attention_mask=..., labels=...)
```

**占位符位置以 LongTensor 傳遞**（每個樣本一個整數位置，-1 表示無占位符），完全相容 HF Trainer 的 `.to(device)` 機制，無需任何 hack。

### 3.4 Backbone 工廠（`model/backbone.py`）

```python
def build_backbone(config: BackboneConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    載入 HF 模型、注冊特殊 token、套用 LoRA，返回 (model, tokenizer)

    特殊 token 注冊：
    - tokenizer.add_special_tokens({"additional_special_tokens": ["[USER_TOKEN]", "[ITEM_TOKEN]"]})
    - model.resize_token_embeddings(len(tokenizer))
    確保占位符被 tokenize 為單一 token，embedding 替換位置唯一確定。
    """
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, ...)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    # 注冊占位符 token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[USER_TOKEN]", "[ITEM_TOKEN]"]})
    model.resize_token_embeddings(len(tokenizer))
    # 套用 LoRA
    lora_config = LoraConfig(r=config.lora_r, ...)
    model = get_peft_model(model, lora_config)
    return model, tokenizer
```

---

## 四、數據層設計

### 4.1 DataCollator（`data/collator.py`）

負責將原始樣本轉換為模型輸入：

1. 根據 `use_collaborative_signal` 選擇對應的 prompt 模板（`prompts.py` 提供兩個版本）
2. Tokenize 後搜尋 `[USER_TOKEN]` 和 `[ITEM_TOKEN]` 的 token ID 位置
3. 非序列模型（MF/LightGCN）的批次中，`seq_history` 鍵**完全不出現**在返回字典中（不是 `None`，而是缺失）；序列模型才包含此鍵
4. 返回字典，所有值均為 Tensor（相容 HF Trainer 的 device 轉移）：

```python
# 非序列模型（MF / LightGCN）
{
    "input_ids": LongTensor,             # (batch, seq_len)
    "attention_mask": LongTensor,        # (batch, seq_len)
    "labels": LongTensor,                # (batch, seq_len)
    "user_ids": LongTensor,              # (batch,)
    "target_item_ids": LongTensor,       # (batch,)
    "user_placeholder_pos": LongTensor,  # (batch,)，找不到時為 -1
    "item_placeholder_pos": LongTensor,  # (batch,)，找不到時為 -1
    # 不含 seq_history 鍵
}

# 序列模型（SASRec / DIN）
{
    ...,  # 同上
    "seq_history": LongTensor,           # (batch, max_history)，pad 為 0
}
```

> **HF Trainer 相容性**：`CoLLMTrainer` 覆寫 `_prepare_inputs()`，在呼叫 `super()` 前先將批次中不存在的鍵過濾掉，`forward()` 對 `seq_history` 使用 `kwargs.get("seq_history", None)` 取值，保持簽名穩定。

### 4.2 Prompt 模板（`data/prompts.py`）

提供兩個版本，消融實驗時直接切換：
- **含占位符版**（完整 CoLLM）：`"User [USER_TOKEN] has watched: {history}. Would user like [ITEM_TOKEN] {item_title}?"`
- **無占位符版**（消融實驗）：`"A user has watched: {history}. Would the user like {item_title}?"`

---

## 五、訓練配置

### 5.1 配置 Dataclass（`training/config.py`）

```python
@dataclass
class RecEncoderConfig:
    encoder_type: str                    # "mf" | "lightgcn" | "sasrec" | "din"
    checkpoint_path: str
    embedding_dim: int = 64
    use_collaborative_signal: bool = True  # False = 消融實驗（純文字模式）

@dataclass
class BackboneConfig:
    model_name_or_path: str              # 預設 "lmsys/vicuna-7b-v1.3"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    load_in_8bit: bool = False

@dataclass
class DataConfig:
    data_path: str
    dataset_type: str                    # "movielens" | "amazon"
    max_history_len: int = 20
    max_seq_len: int = 512

@dataclass
class CoLLMConfig:
    rec: RecEncoderConfig
    backbone: BackboneConfig
    data: DataConfig
    # HF TrainingArguments 在訓練腳本中單獨實例化，不嵌套於此
```

**YAML 載入**：使用 `dacite.from_dict()` 將 `yaml.safe_load()` 的結果轉換為 `CoLLMConfig` 實例：

```python
import yaml, dacite
cfg = dacite.from_dict(CoLLMConfig, yaml.safe_load(open("config.yaml")))
training_args = TrainingArguments(**yaml.safe_load(open("training_args.yaml")))
```

### 5.2 Rec Encoder 預訓練（`scripts/train_rec_encoder.py`）

Rec Encoder 完全獨立於 CoLLM 框架預先訓練：
- **訓練目標**：BPR loss（排名學習）或 BCE loss（二分類）
- **輸入**：用戶-物品交互對（正樣本）+ 負採樣
- **輸出**：保存用戶 embedding matrix 和物品 embedding matrix 的 checkpoint（`.pth`）
- Stage 2 時透過 `RecEncoderConfig.checkpoint_path` 載入，在 CoLLM 訓練過程中**凍結**

### 5.3 兩階段訓練

**Stage 1**（`scripts/train_stage1.py`）：
- 構建 `CoLLMModel(backbone, tokenizer, rec_encoder=None, cie_module=None)`
- `use_collaborative_signal=False`，使用無占位符 prompt 模板
- 只有 LoRA 參數可訓練，backbone 其餘部分凍結
- 訓練目標：讓 LLM 理解推薦任務格式與語言風格

**Stage 2**（`scripts/train_stage2.py`）：
- 載入 Stage 1 checkpoint
- 載入預訓練 Rec Encoder（凍結）
- 初始化 CIE module（可訓練）
- 凍結 backbone（含 LoRA）和 rec encoder（**此為刻意設計**：Stage 1 已賦予 LoRA 推薦任務的語言能力，Stage 2 的目的僅是讓 CIE 投影層學會對齊協同 embedding 與 LLM token 空間，凍結其餘參數可防止遺忘並穩定訓練）
- `use_collaborative_signal=True`，使用含占位符 prompt 模板
- 只有 CIE module 的 `proj` 參數可訓練

---

## 六、消融實驗支持

透過 `RecEncoderConfig.use_collaborative_signal` 控制 prompt 模板選擇：

| 設定 | Prompt 模板 | 占位符 token | CoLLMModel 行為 | 對應論文實驗 |
|------|------------|-------------|----------------|-------------|
| `True` | 含占位符版 | `[USER_TOKEN]`, `[ITEM_TOKEN]` 存在於序列中 | 替換對應位置 embedding | 完整 CoLLM |
| `False` | 無占位符版 | 序列中不含特殊 token | 無需替換，直接送入 backbone | 無 ID 信號消融組 |

切換只需改一個參數，不需修改任何模型代碼。消融模式下（無占位符版 prompt），由於序列中不含特殊 token，collator 找不到對應位置，`user_placeholder_pos` 和 `item_placeholder_pos` 均為 -1，`forward()` 跳過 embedding 替換步驟。

---

## 七、與原始論文的對應

| 原始組件 | 重構後對應 |
|----------|-----------|
| `rec_base_models.py` | `encoders/mf.py`, `lightgcn.py`, `sasrec.py`, `din.py` |
| `rec_model.py` (Rec2Base) | `encoders/base.py` (BaseRecEncoder) |
| `minigpt4rec_v2.py` | `model/collm.py` + `model/cie.py` |
| `modeling_llama.py` | HF 原生 LLaMA（透過 `inputs_embeds`） |
| `runner_base_rec.py` | HF Trainer（`training/trainer.py`） |
| `minigpt4/common/config.py` | `training/config.py`（dataclass + dacite） |
| `rec_dataset.py` | `data/dataset.py` + `data/collator.py` |
| 無（外部訓練） | `scripts/train_rec_encoder.py` |
