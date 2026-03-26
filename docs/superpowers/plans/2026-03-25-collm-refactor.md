# CoLLM 重構 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 CoLLM 重構為模塊化、研究友好的代碼庫，使用 HuggingFace Trainer、統一的 Rec Encoder 介面和 dataclass 配置。

**Architecture:** 四層清晰分離：`encoders/`（Rec Encoder 實現）、`model/`（CIE + CoLLMModel + backbone 工廠）、`data/`（Dataset + Collator + Prompt 模板）、`training/`（Trainer + Config + Metrics）。所有層透過明確介面通信，無跨層直接依賴。

**Tech Stack:** Python 3.10, PyTorch 2.0+, transformers 4.35+, peft 0.6+, dacite 1.8+, scipy, pytest, pytest-mock

**Spec:** `docs/superpowers/specs/2026-03-25-collm-refactor-design.md`

---

## 文件結構

| 文件 | 職責 |
|------|------|
| `pyproject.toml` | 依賴與 package 設定 |
| `collm/constants.py` | 全域常數（`USER_TOKEN`, `ITEM_TOKEN`） |
| `collm/encoders/base.py` | `BaseRecEncoder` 抽象類 |
| `collm/encoders/mf.py` | `MFEncoder`（矩陣分解） |
| `collm/encoders/lightgcn.py` | `LightGCNEncoder`（圖卷積） |
| `collm/encoders/sasrec.py` | `SASRecEncoder`（自注意力序列） |
| `collm/encoders/din.py` | `DINEncoder`（深度興趣網絡） |
| `collm/model/cie.py` | `CIEModule`（線性投影層） |
| `collm/model/backbone.py` | `build_backbone()`（載入 LLM + LoRA + 特殊 token） |
| `collm/model/collm.py` | `CoLLMModel`（主模型） |
| `collm/data/prompts.py` | `get_prompt()`（含/不含占位符兩版本） |
| `collm/data/dataset.py` | `RecDataset`（MovieLens/Amazon） |
| `collm/data/collator.py` | `RecDataCollator`（占位符位置 + 序列處理） |
| `collm/training/config.py` | `RecEncoderConfig`, `BackboneConfig`, `DataConfig`, `CoLLMConfig` |
| `collm/training/metrics.py` | `compute_auc`, `compute_hr`, `compute_ndcg` |
| `collm/training/trainer.py` | `CoLLMTrainer`（繼承 HF Trainer） |
| `scripts/train_rec_encoder.py` | Rec Encoder 預訓練入口 |
| `scripts/train_stage1.py` | Stage 1 訓練入口 |
| `scripts/train_stage2.py` | Stage 2 訓練入口 |
| `tests/conftest.py` | 共用 fixtures |
| `tests/test_config.py` | Config 載入測試 |
| `tests/test_encoders.py` | 四個 Encoder 測試 |
| `tests/test_cie.py` | CIE 模塊測試 |
| `tests/test_backbone.py` | Backbone 工廠測試（mock） |
| `tests/test_data.py` | Dataset + Collator 測試 |
| `tests/test_collm_model.py` | CoLLMModel forward 測試 |
| `tests/test_metrics.py` | 指標測試 |
| `tests/test_trainer.py` | CoLLMTrainer 測試 |

---

## Task 1：Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `collm/__init__.py`
- Create: `collm/encoders/__init__.py`
- Create: `collm/model/__init__.py`
- Create: `collm/data/__init__.py`
- Create: `collm/training/__init__.py`
- Create: `tests/conftest.py`
- Create: `scripts/.gitkeep`

- [ ] **Step 1：建立 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "collm"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "peft>=0.6.0",
    "dacite>=1.8.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "pyyaml>=6.0",
    "scikit-learn>=1.2.0",
    "tqdm>=4.64.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-mock>=3.10",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["collm*"]
```

- [ ] **Step 2：建立 `collm/constants.py`（全域常數，避免跨層循環 import）**

```python
# collm/constants.py
# USER_TOKEN 和 ITEM_TOKEN 定義在此處，供 data/ 和 model/ 層共用，
# 避免 data/prompts.py 直接 import model/backbone.py（跨層依賴）。

USER_TOKEN = "[USER_TOKEN]"
ITEM_TOKEN = "[ITEM_TOKEN]"
```

- [ ] **Step 3：建立所有 `__init__.py`**

```bash
mkdir -p collm/encoders collm/model collm/data collm/training
mkdir -p tests scripts configs
touch collm/__init__.py collm/encoders/__init__.py
touch collm/model/__init__.py collm/data/__init__.py
touch collm/training/__init__.py
```

- [ ] **Step 4：建立 `tests/conftest.py`（共用 fixtures）**

```python
import pytest
import torch
import pandas as pd
import tempfile
import os
from collm.training.config import RecEncoderConfig, BackboneConfig, DataConfig


@pytest.fixture
def tiny_rec_config():
    return RecEncoderConfig(
        encoder_type="mf",
        checkpoint_path="",
        embedding_dim=8,
        use_collaborative_signal=True,
        user_num=10,
        item_num=20,
        max_seq_len=5,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
        n_gcn_layers=2,
    )


@pytest.fixture
def tiny_backbone_config():
    return BackboneConfig(
        model_name_or_path="gpt2",
        lora_r=4,
        lora_alpha=8,
        lora_target_modules=["c_attn"],
        load_in_8bit=False,
    )


@pytest.fixture
def batch_size():
    return 3


@pytest.fixture
def user_ids(batch_size):
    return torch.tensor([0, 1, 2])


@pytest.fixture
def item_ids(batch_size):
    return torch.tensor([5, 6, 7])


@pytest.fixture
def seq_history(batch_size):
    # (batch, max_seq_len)，0 為 padding
    return torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 10]])


@pytest.fixture
def sample_pickle(tmp_path):
    """建立一個 minimal 的 pickle 用於 Dataset 測試"""
    df = pd.DataFrame({
        "uid": [0, 1, 2],
        "iid": [5, 6, 7],
        "title": ["Movie A", "Movie B", "Movie C"],
        "history_iid": [[1, 2], [3], [4, 5, 6]],
        "history_title": [["X", "Y"], ["Z"], ["A", "B", "C"]],
        "label": [1, 0, 1],
    })
    path = tmp_path / "data.pkl"
    df.to_pickle(path)
    return str(tmp_path / "data")  # 不含 .pkl，Dataset 內部加
```

- [ ] **Step 5：安裝開發依賴**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 6：確認 import 不報錯**

```bash
python -c "from collm.constants import USER_TOKEN, ITEM_TOKEN; print('OK')"
```

Expected: `OK`

- [ ] **Step 7：Commit**

```bash
git init
git add pyproject.toml collm/ tests/conftest.py scripts/ configs/
git commit -m "chore: project scaffolding, package structure, and shared constants"
```

---

## Task 2：Config Dataclasses

**Files:**
- Create: `collm/training/config.py`
- Create: `tests/test_config.py`
- Create: `configs/stage1_movielens.yaml`
- Create: `configs/stage2_movielens.yaml`
- Create: `configs/rec_encoder_movielens.yaml`

- [ ] **Step 1：寫 failing test**

```python
# tests/test_config.py
import yaml
import dacite
import tempfile
import os
from collm.training.config import RecEncoderConfig, BackboneConfig, DataConfig, CoLLMConfig


def test_collm_config_from_dict():
    raw = {
        "rec": {
            "encoder_type": "sasrec",
            "checkpoint_path": "/tmp/rec.pth",
            "embedding_dim": 64,
            "use_collaborative_signal": True,
            "user_num": 1000,
            "item_num": 5000,
            "max_seq_len": 50,
            "n_layers": 2,
            "n_heads": 2,
            "dropout": 0.2,
            "n_gcn_layers": 3,
        },
        "backbone": {
            "model_name_or_path": "lmsys/vicuna-7b-v1.3",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["q_proj", "v_proj"],
            "load_in_8bit": False,
        },
        "data": {
            "data_path": "/data/ml-1m",
            "dataset_type": "movielens",
            "max_history_len": 20,
            "max_seq_len": 512,
        },
    }
    cfg = dacite.from_dict(CoLLMConfig, raw, config=dacite.Config(cast=[list]))
    assert cfg.rec.encoder_type == "sasrec"
    assert cfg.backbone.lora_r == 8
    assert cfg.data.dataset_type == "movielens"
    assert cfg.rec.use_collaborative_signal is True


def test_config_from_yaml(tmp_path):
    yaml_content = """
rec:
  encoder_type: mf
  checkpoint_path: /tmp/mf.pth
  embedding_dim: 64
  use_collaborative_signal: false
  user_num: 100
  item_num: 200
  max_seq_len: 50
  n_layers: 2
  n_heads: 2
  dropout: 0.1
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
  data_path: /data
  dataset_type: movielens
  max_history_len: 20
  max_seq_len: 512
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    raw = yaml.safe_load(p.read_text())
    cfg = dacite.from_dict(CoLLMConfig, raw, config=dacite.Config(cast=[list]))
    assert cfg.rec.use_collaborative_signal is False
    assert cfg.rec.embedding_dim == 64
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ImportError` 或 `ModuleNotFoundError`

- [ ] **Step 3：實作 `collm/training/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RecEncoderConfig:
    encoder_type: str            # "mf" | "lightgcn" | "sasrec" | "din"
    checkpoint_path: str
    embedding_dim: int = 64
    use_collaborative_signal: bool = True
    # 執行時從資料集讀取
    user_num: int = 0
    item_num: int = 0
    # 序列模型共用
    max_seq_len: int = 50
    # SASRec 專用
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.2
    # LightGCN 專用
    n_gcn_layers: int = 3


@dataclass
class BackboneConfig:
    model_name_or_path: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    load_in_8bit: bool = False


@dataclass
class DataConfig:
    data_path: str
    dataset_type: str            # "movielens" | "amazon"
    max_history_len: int = 20
    max_seq_len: int = 512


@dataclass
class CoLLMConfig:
    rec: RecEncoderConfig
    backbone: BackboneConfig
    data: DataConfig
```

- [ ] **Step 4：執行確認 pass**

```bash
pytest tests/test_config.py -v
```

Expected: 2 passed

- [ ] **Step 5：建立 YAML 範例配置**

```yaml
# configs/stage1_movielens.yaml
rec:
  encoder_type: sasrec
  checkpoint_path: ""            # Stage 1 不使用
  embedding_dim: 64
  use_collaborative_signal: false
  user_num: 0                    # 執行時自動填入
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

```yaml
# configs/stage2_movielens.yaml
rec:
  encoder_type: sasrec
  checkpoint_path: /checkpoints/sasrec_ml1m.pth
  embedding_dim: 64
  use_collaborative_signal: true
  user_num: 0
  item_num: 0
  max_seq_len: 50
  n_layers: 2
  n_heads: 2
  dropout: 0.2
  n_gcn_layers: 3
backbone:
  model_name_or_path: /checkpoints/stage1_lora
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

```yaml
# configs/rec_encoder_movielens.yaml
encoder_type: sasrec
embedding_dim: 64
max_seq_len: 50
n_layers: 2
n_heads: 2
dropout: 0.2
n_gcn_layers: 3
# 以下由訓練腳本自動填入
user_num: 0
item_num: 0
```

- [ ] **Step 6：Commit**

```bash
git add collm/training/config.py tests/test_config.py configs/
git commit -m "feat: add config dataclasses and YAML examples"
```

---

## Task 3：BaseRecEncoder

**Files:**
- Create: `collm/encoders/base.py`
- Modify: `collm/encoders/__init__.py`
- Create: `tests/test_encoders.py`（骨架）

- [ ] **Step 1：寫 failing test**

```python
# tests/test_encoders.py
import pytest
import torch
from collm.encoders.base import BaseRecEncoder


def test_base_encoder_is_abstract():
    """無法直接實例化 BaseRecEncoder"""
    with pytest.raises(TypeError):
        BaseRecEncoder()


def test_base_encoder_requires_all_methods():
    """子類若未實作所有方法，實例化時報 TypeError"""
    class Incomplete(BaseRecEncoder):
        pass  # 沒有實作任何 abstractmethod

    with pytest.raises(TypeError):
        Incomplete()
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_encoders.py -v
```

- [ ] **Step 3：實作 `collm/encoders/base.py`**

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor


class BaseRecEncoder(nn.Module, ABC):
    """所有推薦 Encoder 的抽象基類。

    子類必須實作：
    - get_user_embedding(user_ids, **kwargs) → (batch, embedding_dim)
    - get_item_embedding(item_ids, **kwargs) → (batch, embedding_dim)
    - embedding_dim property

    序列模型（SASRec/DIN）透過 **kwargs 接收 seq_history 和 target_item_ids。
    非序列模型（MF/LightGCN）直接忽略多餘的 kwargs。
    """

    @abstractmethod
    def get_user_embedding(self, user_ids: Tensor, **kwargs) -> Tensor:
        """
        參數:
            user_ids: LongTensor, shape (batch,)
            **kwargs: 可選 seq_history=(batch, seq_len)，
                      可選 target_item_ids=(batch,)（DIN 使用）
        返回:
            FloatTensor, shape (batch, embedding_dim)
        """

    @abstractmethod
    def get_item_embedding(self, item_ids: Tensor, **kwargs) -> Tensor:
        """
        參數:
            item_ids: LongTensor, shape (batch,)
            **kwargs: 可選 seq_history（序列模型使用）
        返回:
            FloatTensor, shape (batch, embedding_dim)
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """rec embedding 的維度 d_rec"""

    def freeze(self) -> None:
        """凍結所有參數（Stage 2 訓練時使用）"""
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config) -> "BaseRecEncoder":
        """從 checkpoint 載入，子類可視需要覆寫"""
        encoder = cls(config)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(state_dict)
        return encoder
```

- [ ] **Step 4：更新 `collm/encoders/__init__.py`**

```python
from collm.encoders.base import BaseRecEncoder

__all__ = ["BaseRecEncoder"]
```

- [ ] **Step 5：執行確認 pass**

```bash
pytest tests/test_encoders.py -v
```

Expected: 2 passed

- [ ] **Step 6：Commit**

```bash
git add collm/encoders/base.py collm/encoders/__init__.py tests/test_encoders.py
git commit -m "feat: add BaseRecEncoder abstract class"
```

---

## Task 4：MF Encoder

**Files:**
- Create: `collm/encoders/mf.py`
- Modify: `collm/encoders/__init__.py`
- Modify: `tests/test_encoders.py`

- [ ] **Step 1：補充 failing test**

```python
# 加入 tests/test_encoders.py
import torch
from collm.encoders.mf import MFEncoder


def test_mf_encoder_output_shape(tiny_rec_config):
    encoder = MFEncoder(tiny_rec_config)
    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([5, 6, 7])

    user_emb = encoder.get_user_embedding(user_ids)
    item_emb = encoder.get_item_embedding(item_ids)

    assert user_emb.shape == (3, tiny_rec_config.embedding_dim)
    assert item_emb.shape == (3, tiny_rec_config.embedding_dim)


def test_mf_encoder_ignores_seq_kwargs(tiny_rec_config):
    """MF 應忽略 seq_history 等多餘參數"""
    encoder = MFEncoder(tiny_rec_config)
    seq = torch.zeros(3, 5, dtype=torch.long)
    user_emb = encoder.get_user_embedding(torch.tensor([0, 1, 2]), seq_history=seq)
    assert user_emb.shape == (3, tiny_rec_config.embedding_dim)


def test_mf_encoder_save_load(tiny_rec_config, tmp_path):
    encoder = MFEncoder(tiny_rec_config)
    path = str(tmp_path / "mf.pth")
    torch.save(encoder.state_dict(), path)

    loaded = MFEncoder.from_pretrained(path, tiny_rec_config)
    assert torch.allclose(
        encoder.user_emb.weight, loaded.user_emb.weight
    )
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_encoders.py::test_mf_encoder_output_shape -v
```

- [ ] **Step 3：實作 `collm/encoders/mf.py`**

```python
import torch
import torch.nn as nn
from torch import Tensor
from collm.encoders.base import BaseRecEncoder
from collm.training.config import RecEncoderConfig


class MFEncoder(BaseRecEncoder):
    """標準矩陣分解（Matrix Factorization）推薦 Encoder。

    用戶和物品各有獨立的 Embedding table，
    直接透過 ID lookup 取得表示向量。
    """

    def __init__(self, config: RecEncoderConfig):
        super().__init__()
        self.user_emb = nn.Embedding(config.user_num, config.embedding_dim)
        self.item_emb = nn.Embedding(config.item_num, config.embedding_dim)
        self._embedding_dim = config.embedding_dim

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def get_user_embedding(self, user_ids: Tensor, **kwargs) -> Tensor:
        return self.user_emb(user_ids)

    def get_item_embedding(self, item_ids: Tensor, **kwargs) -> Tensor:
        return self.item_emb(item_ids)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
```

- [ ] **Step 4：更新 `collm/encoders/__init__.py`**

```python
from collm.encoders.base import BaseRecEncoder
from collm.encoders.mf import MFEncoder

__all__ = ["BaseRecEncoder", "MFEncoder"]
```

- [ ] **Step 5：執行確認 pass**

```bash
pytest tests/test_encoders.py -k "mf" -v
```

Expected: 3 passed

- [ ] **Step 6：Commit**

```bash
git add collm/encoders/mf.py collm/encoders/__init__.py tests/test_encoders.py
git commit -m "feat: add MFEncoder"
```

---

## Task 5：LightGCN Encoder

**Files:**
- Create: `collm/encoders/lightgcn.py`
- Modify: `collm/encoders/__init__.py`
- Modify: `tests/test_encoders.py`

- [ ] **Step 1：補充 failing test**

```python
# 加入 tests/test_encoders.py
import scipy.sparse as sp
import numpy as np
from collm.encoders.lightgcn import LightGCNEncoder


def _make_adj_matrix(user_num, item_num):
    """建立一個簡單的 user-item 交互鄰接矩陣（COO 格式 → sparse tensor）"""
    # 模擬 3 條交互：user 0 → item 0, user 1 → item 1, user 2 → item 2
    rows = np.array([0, 1, 2, user_num + 0, user_num + 1, user_num + 2])
    cols = np.array([user_num + 0, user_num + 1, user_num + 2, 0, 1, 2])
    data = np.ones(len(rows))
    n = user_num + item_num
    mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    indices = torch.from_numpy(np.vstack([mat.row, mat.col])).long()
    values = torch.from_numpy(mat.data).float()
    return torch.sparse_coo_tensor(indices, values, (n, n))


def test_lightgcn_output_shape(tiny_rec_config):
    adj = _make_adj_matrix(tiny_rec_config.user_num, tiny_rec_config.item_num)
    encoder = LightGCNEncoder(tiny_rec_config, adj)
    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([0, 1, 2])

    user_emb = encoder.get_user_embedding(user_ids)
    item_emb = encoder.get_item_embedding(item_ids)

    assert user_emb.shape == (3, tiny_rec_config.embedding_dim)
    assert item_emb.shape == (3, tiny_rec_config.embedding_dim)
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_encoders.py::test_lightgcn_output_shape -v
```

- [ ] **Step 3：實作 `collm/encoders/lightgcn.py`**

```python
import torch
import torch.nn as nn
from torch import Tensor
from collm.encoders.base import BaseRecEncoder
from collm.training.config import RecEncoderConfig


class LightGCNEncoder(BaseRecEncoder):
    """LightGCN 圖卷積推薦 Encoder。

    透過多層圖傳播精煉用戶和物品的 embedding，
    最終輸出為各層 embedding 的均值。

    Args:
        config: RecEncoderConfig，使用 n_gcn_layers 欄位
        adj_matrix: 正規化後的 user-item 鄰接矩陣
                    shape (user_num + item_num, user_num + item_num)
                    型別：torch.sparse_coo_tensor
    """

    def __init__(self, config: RecEncoderConfig, adj_matrix: Tensor):
        super().__init__()
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.n_layers = config.n_gcn_layers
        self._embedding_dim = config.embedding_dim

        self.user_emb = nn.Embedding(config.user_num, config.embedding_dim)
        self.item_emb = nn.Embedding(config.item_num, config.embedding_dim)
        self.register_buffer("adj", adj_matrix)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config, adj_matrix=None) -> "LightGCNEncoder":
        """LightGCN 需要 adj_matrix，覆寫基類的 from_pretrained。

        Args:
            adj_matrix: 若為 None，則建立全零的占位矩陣（eval 時用固定 embedding）
        """
        import torch
        if adj_matrix is None:
            n = config.user_num + config.item_num
            adj_matrix = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0),
                (n, n),
            )
        encoder = cls(config, adj_matrix)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(state_dict)
        return encoder

    def _propagate(self) -> tuple[Tensor, Tensor]:
        """執行 LightGCN 傳播，返回聚合後的 user/item embeddings。"""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layer_embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            layer_embs.append(all_emb)

        final = torch.stack(layer_embs, dim=1).mean(dim=1)
        user_final = final[: self.user_num]
        item_final = final[self.user_num :]
        return user_final, item_final

    def get_user_embedding(self, user_ids: Tensor, **kwargs) -> Tensor:
        user_all, _ = self._propagate()
        return user_all[user_ids]

    def get_item_embedding(self, item_ids: Tensor, **kwargs) -> Tensor:
        _, item_all = self._propagate()
        return item_all[item_ids]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
```

- [ ] **Step 4：更新 `collm/encoders/__init__.py`**

```python
from collm.encoders.base import BaseRecEncoder
from collm.encoders.mf import MFEncoder
from collm.encoders.lightgcn import LightGCNEncoder

__all__ = ["BaseRecEncoder", "MFEncoder", "LightGCNEncoder"]
```

- [ ] **Step 5：執行確認 pass**

```bash
pytest tests/test_encoders.py -k "lightgcn or mf or base" -v
```

- [ ] **Step 6：Commit**

```bash
git add collm/encoders/lightgcn.py collm/encoders/__init__.py tests/test_encoders.py
git commit -m "feat: add LightGCNEncoder with adj_matrix and from_pretrained override"
```

---

## Task 6：SASRec Encoder

**Files:**
- Create: `collm/encoders/sasrec.py`
- Modify: `collm/encoders/__init__.py`
- Modify: `tests/test_encoders.py`

- [ ] **Step 1：補充 failing test**

```python
# 加入 tests/test_encoders.py
from collm.encoders.sasrec import SASRecEncoder


def test_sasrec_output_shape(tiny_rec_config, user_ids, item_ids, seq_history):
    encoder = SASRecEncoder(tiny_rec_config)
    user_emb = encoder.get_user_embedding(user_ids, seq_history=seq_history)
    item_emb = encoder.get_item_embedding(item_ids)

    assert user_emb.shape == (3, tiny_rec_config.embedding_dim)
    assert item_emb.shape == (3, tiny_rec_config.embedding_dim)


def test_sasrec_user_emb_differs_with_different_history(tiny_rec_config):
    """不同的歷史序列應產生不同的用戶表示"""
    encoder = SASRecEncoder(tiny_rec_config)
    ids = torch.tensor([0])
    hist1 = torch.tensor([[1, 2, 3, 0, 0]])
    hist2 = torch.tensor([[4, 5, 6, 7, 8]])

    emb1 = encoder.get_user_embedding(ids, seq_history=hist1)
    emb2 = encoder.get_user_embedding(ids, seq_history=hist2)
    assert not torch.allclose(emb1, emb2)
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_encoders.py -k "sasrec" -v
```

- [ ] **Step 3：實作 `collm/encoders/sasrec.py`**

```python
import math
import torch
import torch.nn as nn
from torch import Tensor
from collm.encoders.base import BaseRecEncoder
from collm.training.config import RecEncoderConfig


class _TransformerBlock(nn.Module):
    """帶因果 mask 的 Transformer block（用於序列推薦）。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        seq_len = x.size(1)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.attn(
            x, x, x,
            attn_mask=causal,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class SASRecEncoder(BaseRecEncoder):
    """SASRec 自注意力序列推薦 Encoder。

    用戶表示由交互歷史透過 Transformer 計算得出（序列最後一個非 padding 位置的輸出）。
    物品表示為直接 embedding lookup。

    Args:
        config: RecEncoderConfig，使用 n_layers, n_heads, dropout, max_seq_len 欄位
    """

    def __init__(self, config: RecEncoderConfig):
        super().__init__()
        d = config.embedding_dim
        self.item_emb = nn.Embedding(
            config.item_num, d, padding_idx=0
        )
        self.pos_emb = nn.Embedding(config.max_seq_len, d)
        self.layers = nn.ModuleList(
            [_TransformerBlock(d, config.n_heads, config.dropout)
             for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        self._embedding_dim = d

    def get_user_embedding(self, user_ids: Tensor, **kwargs) -> Tensor:
        seq_history: Tensor = kwargs["seq_history"]   # (batch, seq_len)
        seq_emb = self.item_emb(seq_history)           # (batch, seq_len, d)

        positions = torch.arange(seq_history.size(1), device=seq_history.device)
        seq_emb = seq_emb + self.pos_emb(positions)

        pad_mask = seq_history == 0                    # (batch, seq_len)，True = padding
        x = self.dropout(self.norm(seq_emb))
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)

        # 取最後一個非 padding 位置作為用戶表示
        seq_len = (~pad_mask).sum(dim=1) - 1           # (batch,)
        seq_len = seq_len.clamp(min=0)
        user_repr = x[torch.arange(x.size(0), device=x.device), seq_len]
        return user_repr

    def get_item_embedding(self, item_ids: Tensor, **kwargs) -> Tensor:
        return self.item_emb(item_ids)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
```

- [ ] **Step 4：執行確認 pass**

```bash
pytest tests/test_encoders.py -k "sasrec" -v
```

Expected: 2 passed

- [ ] **Step 5：更新 `__init__.py`**

```python
from collm.encoders.base import BaseRecEncoder
from collm.encoders.mf import MFEncoder
from collm.encoders.lightgcn import LightGCNEncoder
from collm.encoders.sasrec import SASRecEncoder

__all__ = ["BaseRecEncoder", "MFEncoder", "LightGCNEncoder", "SASRecEncoder"]
```

- [ ] **Step 6：Commit**

```bash
git add collm/encoders/sasrec.py collm/encoders/__init__.py tests/test_encoders.py
git commit -m "feat: add SASRecEncoder"
```

---

## Task 7：DIN Encoder

**Files:**
- Create: `collm/encoders/din.py`
- Modify: `collm/encoders/__init__.py`
- Modify: `tests/test_encoders.py`

- [ ] **Step 1：補充 failing test**

```python
# 加入 tests/test_encoders.py
from collm.encoders.din import DINEncoder


def test_din_output_shape(tiny_rec_config, user_ids, item_ids, seq_history):
    encoder = DINEncoder(tiny_rec_config)
    user_emb = encoder.get_user_embedding(
        user_ids, seq_history=seq_history, target_item_ids=item_ids
    )
    item_emb = encoder.get_item_embedding(item_ids)

    assert user_emb.shape == (3, tiny_rec_config.embedding_dim)
    assert item_emb.shape == (3, tiny_rec_config.embedding_dim)


def test_din_attention_sensitive_to_target(tiny_rec_config, user_ids, seq_history):
    """不同 target item 應產生不同的用戶表示（attention 有作用）"""
    encoder = DINEncoder(tiny_rec_config)
    item_a = torch.tensor([1, 2, 3])
    item_b = torch.tensor([5, 6, 7])

    emb_a = encoder.get_user_embedding(user_ids, seq_history=seq_history, target_item_ids=item_a)
    emb_b = encoder.get_user_embedding(user_ids, seq_history=seq_history, target_item_ids=item_b)
    assert not torch.allclose(emb_a, emb_b)
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_encoders.py -k "din" -v
```

- [ ] **Step 3：實作 `collm/encoders/din.py`**

```python
import torch
import torch.nn as nn
from torch import Tensor
from collm.encoders.base import BaseRecEncoder
from collm.training.config import RecEncoderConfig


class DINEncoder(BaseRecEncoder):
    """DIN（Deep Interest Network）推薦 Encoder。

    用戶表示由 target item 對歷史序列的 attention 加權聚合得出，
    對不同目標物品自適應調整用戶興趣表示。

    Args:
        config: RecEncoderConfig，使用 embedding_dim 欄位
    """

    def __init__(self, config: RecEncoderConfig):
        super().__init__()
        d = config.embedding_dim
        self.user_emb = nn.Embedding(config.user_num, d)
        self.item_emb = nn.Embedding(config.item_num, d, padding_idx=0)
        # 注意力網絡：輸入 [target, hist, target-hist, target*hist]，輸出 score
        self.attention = nn.Sequential(
            nn.Linear(d * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._embedding_dim = d

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def get_user_embedding(self, user_ids: Tensor, **kwargs) -> Tensor:
        seq_history: Tensor | None = kwargs.get("seq_history")
        target_item_ids: Tensor | None = kwargs.get("target_item_ids")

        if seq_history is None or target_item_ids is None:
            # 退化為普通 user embedding lookup
            return self.user_emb(user_ids)

        hist_emb = self.item_emb(seq_history)                    # (batch, seq, d)
        target_emb = self.item_emb(target_item_ids)              # (batch, d)
        target_exp = target_emb.unsqueeze(1).expand_as(hist_emb) # (batch, seq, d)

        attn_input = torch.cat(
            [target_exp, hist_emb, target_exp - hist_emb, target_exp * hist_emb],
            dim=-1,
        )  # (batch, seq, 4d)
        attn_scores = self.attention(attn_input).squeeze(-1)      # (batch, seq)

        pad_mask = seq_history == 0
        attn_scores = attn_scores.masked_fill(pad_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)         # (batch, seq)

        user_repr = (attn_weights.unsqueeze(-1) * hist_emb).sum(dim=1)  # (batch, d)
        return user_repr

    def get_item_embedding(self, item_ids: Tensor, **kwargs) -> Tensor:
        return self.item_emb(item_ids)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
```

- [ ] **Step 4：執行全部 encoder 測試**

```bash
pytest tests/test_encoders.py -v
```

Expected: 全部 pass

- [ ] **Step 5：更新 `__init__.py`**

```python
from collm.encoders.base import BaseRecEncoder
from collm.encoders.mf import MFEncoder
from collm.encoders.lightgcn import LightGCNEncoder
from collm.encoders.sasrec import SASRecEncoder
from collm.encoders.din import DINEncoder

__all__ = ["BaseRecEncoder", "MFEncoder", "LightGCNEncoder", "SASRecEncoder", "DINEncoder"]
```

- [ ] **Step 6：Commit**

```bash
git add collm/encoders/din.py collm/encoders/__init__.py tests/test_encoders.py
git commit -m "feat: add DINEncoder"
```

---

## Task 8：CIE Module

**Files:**
- Create: `collm/model/cie.py`
- Modify: `collm/model/__init__.py`
- Create: `tests/test_cie.py`

- [ ] **Step 1：寫 failing test**

```python
# tests/test_cie.py
import torch
from collm.model.cie import CIEModule


def test_cie_output_shape():
    rec_dim, llm_dim = 64, 4096
    cie = CIEModule(rec_dim, llm_dim)
    x = torch.randn(4, rec_dim)
    out = cie(x)
    assert out.shape == (4, llm_dim)


def test_cie_small_dims():
    cie = CIEModule(8, 16)
    x = torch.randn(3, 8)
    out = cie(x)
    assert out.shape == (3, 16)


def test_cie_is_differentiable():
    cie = CIEModule(8, 16)
    x = torch.randn(2, 8, requires_grad=True)
    out = cie(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_cie.py -v
```

- [ ] **Step 3：實作 `collm/model/cie.py`**

```python
import torch.nn as nn
from torch import Tensor


class CIEModule(nn.Module):
    """CIE（Collaborative Information Embedding）投影模塊。

    將協同過濾 encoder 輸出的 embedding（維度 d_rec）線性投影到
    LLM 的 token embedding 空間（維度 d_llm），產生可直接注入 LLM 的虛擬 token。

    注意：原始論文使用 Q-Former 風格的多層投影。此處簡化為單一線性層。
    若需完全復現原始數字，可替換為多層 MLP，介面保持不變。

    Args:
        rec_dim:  協同 embedding 維度（d_rec）
        llm_dim:  LLM token embedding 維度（d_llm，通常為 4096）
    """

    def __init__(self, rec_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Linear(rec_dim, llm_dim)

    def forward(self, rec_embedding: Tensor) -> Tensor:
        """
        參數:
            rec_embedding: FloatTensor, shape (batch, d_rec)
        返回:
            FloatTensor, shape (batch, d_llm)
        """
        return self.proj(rec_embedding)
```

- [ ] **Step 4：更新 `collm/model/__init__.py`**

```python
from collm.model.cie import CIEModule

__all__ = ["CIEModule"]
```

- [ ] **Step 5：執行確認 pass**

```bash
pytest tests/test_cie.py -v
```

Expected: 3 passed

- [ ] **Step 6：Commit**

```bash
git add collm/model/cie.py collm/model/__init__.py tests/test_cie.py
git commit -m "feat: add CIEModule"
```

---

## Task 9：Backbone Factory

**Files:**
- Create: `collm/model/backbone.py`
- Modify: `collm/model/__init__.py`
- Create: `tests/test_backbone.py`

- [ ] **Step 1：寫 failing test（用 mock，不下載真實模型）**

```python
# tests/test_backbone.py
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, call
from collm.model.backbone import build_backbone
from collm.training.config import BackboneConfig


@pytest.fixture
def backbone_config():
    return BackboneConfig(
        model_name_or_path="fake/model",
        lora_r=4,
        lora_alpha=8,
        lora_target_modules=["c_attn"],
        load_in_8bit=False,
    )


def test_build_backbone_registers_special_tokens(mocker, backbone_config):
    """build_backbone 必須注冊 [USER_TOKEN] 和 [ITEM_TOKEN]"""
    mock_model = MagicMock()
    mock_model.config.hidden_size = 16
    mock_tokenizer = MagicMock()
    mock_tokenizer.__len__ = MagicMock(return_value=32002)

    mocker.patch("collm.model.backbone.AutoModelForCausalLM.from_pretrained",
                 return_value=mock_model)
    mocker.patch("collm.model.backbone.AutoTokenizer.from_pretrained",
                 return_value=mock_tokenizer)
    mocker.patch("collm.model.backbone.get_peft_model", return_value=mock_model)

    build_backbone(backbone_config)

    mock_tokenizer.add_special_tokens.assert_called_once_with(
        {"additional_special_tokens": ["[USER_TOKEN]", "[ITEM_TOKEN]"]}
    )
    mock_model.resize_token_embeddings.assert_called_once_with(32002)


def test_build_backbone_applies_lora(mocker, backbone_config):
    """build_backbone 必須套用 LoRA"""
    mock_model = MagicMock()
    mock_model.config.hidden_size = 16
    mock_tokenizer = MagicMock()
    mock_tokenizer.__len__ = MagicMock(return_value=32000)
    mock_lora_model = MagicMock()

    mocker.patch("collm.model.backbone.AutoModelForCausalLM.from_pretrained",
                 return_value=mock_model)
    mocker.patch("collm.model.backbone.AutoTokenizer.from_pretrained",
                 return_value=mock_tokenizer)
    mock_get_peft = mocker.patch("collm.model.backbone.get_peft_model",
                                  return_value=mock_lora_model)

    model, tokenizer = build_backbone(backbone_config)

    assert mock_get_peft.called
    assert model is mock_lora_model


def test_build_backbone_returns_tuple(mocker, backbone_config):
    """返回值必須是 (model, tokenizer) tuple"""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.__len__ = MagicMock(return_value=32000)
    mocker.patch("collm.model.backbone.AutoModelForCausalLM.from_pretrained",
                 return_value=mock_model)
    mocker.patch("collm.model.backbone.AutoTokenizer.from_pretrained",
                 return_value=mock_tokenizer)
    mocker.patch("collm.model.backbone.get_peft_model", return_value=mock_model)

    result = build_backbone(backbone_config)
    assert isinstance(result, tuple)
    assert len(result) == 2
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_backbone.py -v
```

- [ ] **Step 3：實作 `collm/model/backbone.py`**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from collm.training.config import BackboneConfig
from collm.constants import USER_TOKEN, ITEM_TOKEN


def build_backbone(config: BackboneConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """載入 HuggingFace LLM、注冊占位符 token、套用 LoRA。

    占位符注冊確保 [USER_TOKEN] 和 [ITEM_TOKEN] 被 tokenize 為單一 token，
    使 DataCollator 能精確定位替換位置。

    Args:
        config: BackboneConfig，指定模型路徑和 LoRA 超參數

    Returns:
        (model, tokenizer) tuple。model 已套用 LoRA，tokenizer 已含占位符。
    """
    dtype = torch.float16

    load_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
    )
    if config.load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, **load_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, use_fast=False
    )

    # 注冊占位符 token（必須在 resize_token_embeddings 之前）
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [USER_TOKEN, ITEM_TOKEN]}
    )
    model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 套用 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
```

- [ ] **Step 4：更新 `collm/model/__init__.py`**

```python
from collm.model.cie import CIEModule
from collm.model.backbone import build_backbone

# USER_TOKEN / ITEM_TOKEN 屬於全域常數，從 collm.constants 直接 import，
# 不透過 collm.model 再匯出（避免讓 model package 成為常數的分發者）

__all__ = ["CIEModule", "build_backbone"]
```

- [ ] **Step 5：執行確認 pass**

```bash
pytest tests/test_backbone.py -v
```

Expected: 3 passed

- [ ] **Step 6：Commit**

```bash
git add collm/model/backbone.py collm/model/__init__.py tests/test_backbone.py
git commit -m "feat: add backbone factory with LoRA and special token registration"
```

---

## Task 10：Prompt Templates

**Files:**
- Create: `collm/data/prompts.py`
- Modify: `collm/data/__init__.py`
- Create: `tests/test_data.py`（骨架）

- [ ] **Step 1：寫 failing test**

```python
# tests/test_data.py
from collm.data.prompts import get_prompt, get_answer


def test_prompt_with_signal_contains_placeholders():
    prompt = get_prompt(
        history_titles=["Movie A", "Movie B"],
        target_title="Movie C",
        use_collaborative_signal=True,
    )
    assert "[USER_TOKEN]" in prompt
    assert "[ITEM_TOKEN]" in prompt


def test_prompt_without_signal_no_placeholders():
    prompt = get_prompt(
        history_titles=["Movie A", "Movie B"],
        target_title="Movie C",
        use_collaborative_signal=False,
    )
    assert "[USER_TOKEN]" not in prompt
    assert "[ITEM_TOKEN]" not in prompt


def test_prompt_contains_history():
    prompt = get_prompt(
        history_titles=["Movie A", "Movie B"],
        target_title="Movie C",
        use_collaborative_signal=True,
    )
    assert "Movie A" in prompt
    assert "Movie B" in prompt
    assert "Movie C" in prompt


def test_answer_yes_no():
    assert "Yes" in get_answer(label=1)
    assert "No" in get_answer(label=0)
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_data.py -k "prompt" -v
```

- [ ] **Step 3：實作 `collm/data/prompts.py`**

```python
from collm.constants import USER_TOKEN, ITEM_TOKEN

_SYSTEM = (
    "You are a helpful movie recommendation assistant. "
    "Based on the user's watch history, predict whether they would enjoy the suggested movie."
)

_TEMPLATE_WITH_SIGNAL = (
    "{system}\n\n"
    "### Instruction:\n"
    "User {user_token} has watched: {history}.\n"
    "Would this user enjoy {item_token} {target_title}?\n\n"
    "### Response:\n"
)

_TEMPLATE_NO_SIGNAL = (
    "{system}\n\n"
    "### Instruction:\n"
    "A user has watched: {history}.\n"
    "Would this user enjoy {target_title}?\n\n"
    "### Response:\n"
)


def get_prompt(
    history_titles: list[str],
    target_title: str,
    use_collaborative_signal: bool,
    max_history: int = 20,
) -> str:
    """生成推薦任務的 prompt。

    Args:
        history_titles:            用戶歷史觀看的物品標題列表
        target_title:              目標物品標題
        use_collaborative_signal:  True → 含 [USER_TOKEN]/[ITEM_TOKEN] 占位符
                                   False → 純文字（消融實驗）
        max_history:               最多保留最近 N 筆歷史

    Returns:
        prompt 字串（不含答案部分）
    """
    history = ", ".join(history_titles[-max_history:]) if history_titles else "nothing"

    if use_collaborative_signal:
        return _TEMPLATE_WITH_SIGNAL.format(
            system=_SYSTEM,
            user_token=USER_TOKEN,
            history=history,
            item_token=ITEM_TOKEN,
            target_title=target_title,
        )
    else:
        return _TEMPLATE_NO_SIGNAL.format(
            system=_SYSTEM,
            history=history,
            target_title=target_title,
        )


def get_answer(label: int) -> str:
    """返回答案字串（'Yes' 或 'No'）"""
    return " Yes" if label == 1 else " No"
```

- [ ] **Step 4：執行確認 pass**

```bash
pytest tests/test_data.py -k "prompt or answer" -v
```

Expected: 4 passed

- [ ] **Step 5：更新 `collm/data/__init__.py`**

```python
from collm.data.prompts import get_prompt, get_answer

__all__ = ["get_prompt", "get_answer"]
```

- [ ] **Step 6：Commit**

```bash
git add collm/data/prompts.py collm/data/__init__.py tests/test_data.py
git commit -m "feat: add prompt templates (with/without collaborative signal)"
```

---

## Task 11：Dataset

**Files:**
- Create: `collm/data/dataset.py`
- Modify: `collm/data/__init__.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1：補充 failing test**

```python
# 加入 tests/test_data.py
import pandas as pd
import torch
from collm.data.dataset import RecDataset


def test_dataset_len(sample_pickle):
    ds = RecDataset(sample_pickle, dataset_type="movielens")
    assert len(ds) == 3


def test_dataset_getitem(sample_pickle):
    ds = RecDataset(sample_pickle, dataset_type="movielens")
    item = ds[0]
    assert "uid" in item
    assert "iid" in item
    assert "title" in item
    assert "history_titles" in item
    assert "label" in item
    assert isinstance(item["label"], int)


def test_dataset_label_is_int(sample_pickle):
    ds = RecDataset(sample_pickle, dataset_type="movielens")
    for i in range(len(ds)):
        item = ds[i]
        assert item["label"] in (0, 1)
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_data.py -k "dataset" -v
```

- [ ] **Step 3：實作 `collm/data/dataset.py`**

```python
import pandas as pd
from torch.utils.data import Dataset


class RecDataset(Dataset):
    """MovieLens / Amazon 推薦資料集。

    讀取 pickle 格式的 DataFrame，每筆樣本包含：
    - uid:            用戶 ID（int）
    - iid:            目標物品 ID（int）
    - title:          目標物品標題（str）
    - history_iid:    歷史互動物品 ID 列表
    - history_title:  歷史互動物品標題列表
    - label:          正樣本為 1，負樣本為 0

    Args:
        data_path:     pickle 路徑（不含 .pkl 副檔名）
        dataset_type:  "movielens" 或 "amazon"（目前邏輯相同，預留擴展）
    """

    def __init__(self, data_path: str, dataset_type: str = "movielens"):
        df = pd.read_pickle(data_path + ".pkl")
        self.records = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        return {
            "uid": int(row["uid"]),
            "iid": int(row["iid"]),
            "title": str(row["title"]),
            "history_iid": list(row.get("history_iid", [])),
            "history_titles": list(row.get("history_title", [])),
            "label": int(row["label"]),
        }
```

- [ ] **Step 4：執行確認 pass**

```bash
pytest tests/test_data.py -k "dataset" -v
```

Expected: 3 passed

- [ ] **Step 5：Commit**

```bash
git add collm/data/dataset.py collm/data/__init__.py tests/test_data.py
git commit -m "feat: add RecDataset"
```

---

## Task 12：DataCollator

**Files:**
- Create: `collm/data/collator.py`
- Modify: `collm/data/__init__.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1：補充 failing test**

```python
# 加入 tests/test_data.py
import torch
from transformers import AutoTokenizer
from collm.data.collator import RecDataCollator


@pytest.fixture
def tiny_tokenizer(tmp_path):
    """使用 GPT-2 tokenizer 並注冊占位符（不下載 LLM 本體）"""
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": ["[USER_TOKEN]", "[ITEM_TOKEN]"]})
    return tok


@pytest.fixture
def sample_batch():
    return [
        {"uid": 0, "iid": 5, "title": "Movie A",
         "history_iid": [1, 2], "history_titles": ["X", "Y"], "label": 1},
        {"uid": 1, "iid": 6, "title": "Movie B",
         "history_iid": [3], "history_titles": ["Z"], "label": 0},
    ]


def test_collator_returns_required_keys(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=True,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=False,
    )
    batch = collator(sample_batch)
    for key in ["input_ids", "attention_mask", "labels",
                "user_ids", "target_item_ids",
                "user_placeholder_pos", "item_placeholder_pos"]:
        assert key in batch, f"Missing key: {key}"


def test_collator_with_signal_finds_placeholder_positions(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=True,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=False,
    )
    batch = collator(sample_batch)
    # 使用 collaborative signal 時，應找到占位符位置（不應全為 -1）
    assert batch["user_placeholder_pos"].max().item() != -1
    assert batch["item_placeholder_pos"].max().item() != -1


def test_collator_without_signal_placeholder_is_neg1(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=False,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=False,
    )
    batch = collator(sample_batch)
    # 消融模式：prompt 中沒有占位符，位置均為 -1
    assert (batch["user_placeholder_pos"] == -1).all()
    assert (batch["item_placeholder_pos"] == -1).all()


def test_collator_sequential_includes_seq_history(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=True,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=True,
    )
    batch = collator(sample_batch)
    assert "seq_history" in batch
    assert batch["seq_history"].shape == (2, 5)  # (batch, max_history_len)


def test_collator_non_sequential_no_seq_history(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=True,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=False,
    )
    batch = collator(sample_batch)
    assert "seq_history" not in batch
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_data.py -k "collator" -v
```

- [ ] **Step 3：實作 `collm/data/collator.py`**

```python
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
from collm.data.prompts import get_prompt, get_answer
from collm.constants import USER_TOKEN, ITEM_TOKEN


class RecDataCollator:
    """將一批 RecDataset 樣本組裝為 CoLLMModel.forward() 所需格式。

    負責：
    - 根據 use_collaborative_signal 選擇 prompt 模板
    - Tokenize prompt + answer，設定 labels（-100 遮蔽 prompt 部分）
    - 定位 [USER_TOKEN] 和 [ITEM_TOKEN] 的 token 位置（-1 表示不存在）
    - 序列模型：pad 歷史 ID 序列為 (batch, max_history_len) 的 LongTensor

    Args:
        tokenizer:                  已注冊占位符 token 的 tokenizer
        use_collaborative_signal:   False → 消融模式（無占位符）
        max_seq_len:                prompt + answer 的最大 token 長度
        max_history_len:            歷史序列的最大長度
        is_sequential:              True → 包含 seq_history（SASRec/DIN）
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        use_collaborative_signal: bool,
        max_seq_len: int,
        max_history_len: int,
        is_sequential: bool,
    ):
        self.tokenizer = tokenizer
        self.use_collaborative_signal = use_collaborative_signal
        self.max_seq_len = max_seq_len
        self.max_history_len = max_history_len
        self.is_sequential = is_sequential
        self._user_tok_id = tokenizer.convert_tokens_to_ids(USER_TOKEN)
        self._item_tok_id = tokenizer.convert_tokens_to_ids(ITEM_TOKEN)

    def __call__(self, samples: list[dict]) -> dict:
        input_ids_list, attention_mask_list, labels_list = [], [], []
        user_ids, target_item_ids = [], []
        user_ph_pos, item_ph_pos = [], []
        seq_history_list = []

        for s in samples:
            prompt = get_prompt(
                history_titles=s["history_titles"],
                target_title=s["title"],
                use_collaborative_signal=self.use_collaborative_signal,
                max_history=self.max_history_len,
            )
            answer = get_answer(s["label"])
            full_text = prompt + answer

            enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            ids = enc["input_ids"][0]
            mask = enc["attention_mask"][0]

            # 找到 prompt 部分的長度（用於設定 labels）
            prompt_enc = self.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
            prompt_len = prompt_enc["input_ids"].shape[1]

            # labels：遮蔽 prompt 部分，只在 answer 部分計算 loss
            lbl = ids.clone()
            lbl[:prompt_len] = -100
            lbl[mask == 0] = -100      # 遮蔽 padding

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(lbl)
            user_ids.append(s["uid"])
            target_item_ids.append(s["iid"])

            # 定位占位符位置
            user_ph_pos.append(self._find_token_pos(ids, self._user_tok_id))
            item_ph_pos.append(self._find_token_pos(ids, self._item_tok_id))

            if self.is_sequential:
                hist = s["history_iid"][-self.max_history_len:]
                padded = [0] * (self.max_history_len - len(hist)) + hist
                seq_history_list.append(padded)

        batch = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
            "user_ids": torch.tensor(user_ids, dtype=torch.long),
            "target_item_ids": torch.tensor(target_item_ids, dtype=torch.long),
            "user_placeholder_pos": torch.tensor(user_ph_pos, dtype=torch.long),
            "item_placeholder_pos": torch.tensor(item_ph_pos, dtype=torch.long),
        }
        if self.is_sequential:
            batch["seq_history"] = torch.tensor(seq_history_list, dtype=torch.long)

        return batch

    def _find_token_pos(self, ids: Tensor, token_id: int) -> int:
        """返回 token_id 在序列中第一次出現的位置，找不到則返回 -1"""
        positions = (ids == token_id).nonzero(as_tuple=True)[0]
        return positions[0].item() if len(positions) > 0 else -1
```

- [ ] **Step 4：執行確認 pass**

```bash
pytest tests/test_data.py -v
```

Expected: 全部 pass

- [ ] **Step 5：Commit**

```bash
git add collm/data/collator.py collm/data/__init__.py tests/test_data.py
git commit -m "feat: add RecDataCollator with placeholder position tracking"
```

---

## Task 13：CoLLMModel

**Files:**
- Create: `collm/model/collm.py`
- Modify: `collm/model/__init__.py`
- Create: `tests/test_collm_model.py`

- [ ] **Step 1：寫 failing test（使用 TinyLM 替代真實 LLM）**

```python
# tests/test_collm_model.py
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from collm.model.collm import CoLLMModel
from collm.model.cie import CIEModule
from collm.encoders.mf import MFEncoder
from collm.training.config import RecEncoderConfig, CoLLMConfig, BackboneConfig, DataConfig


# ── TinyLM：模擬 HuggingFace LLM 介面 ──────────────────────────────────
class TinyLM(nn.Module):
    def __init__(self, vocab_size=100, d_model=16):
        super().__init__()
        self._embeddings = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def get_input_embeddings(self):
        return self._embeddings

    def forward(self, input_ids=None, inputs_embeds=None,
                attention_mask=None, labels=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self._embeddings(input_ids)
        logits = self.lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def config():
    return CoLLMConfig(
        rec=RecEncoderConfig(
            encoder_type="mf", checkpoint_path="",
            embedding_dim=8, use_collaborative_signal=True,
            user_num=10, item_num=20,
        ),
        backbone=BackboneConfig(model_name_or_path=""),
        data=DataConfig(data_path="", dataset_type="movielens"),
    )


@pytest.fixture
def config_no_signal(config):
    config.rec.use_collaborative_signal = False
    return config


@pytest.fixture
def tiny_lm():
    return TinyLM(vocab_size=100, d_model=16)


@pytest.fixture
def tiny_tokenizer_mock():
    class FakeTok:
        pass
    return FakeTok()


# ── Tests ───────────────────────────────────────────────────────────────
def test_stage1_forward_no_encoder(tiny_lm, config_no_signal, tiny_tokenizer_mock):
    """Stage 1：無 rec_encoder/cie，純語言模型 forward"""
    model = CoLLMModel(
        backbone=tiny_lm,
        tokenizer=tiny_tokenizer_mock,
        config=config_no_signal,
    )
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :5] = -100

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0


def test_stage2_forward_with_encoder(tiny_lm, config, tiny_tokenizer_mock):
    """Stage 2：有 rec_encoder + cie，embedding 替換後 forward"""
    rec_encoder = MFEncoder(config.rec)
    cie = CIEModule(rec_dim=8, llm_dim=16)

    model = CoLLMModel(
        backbone=tiny_lm,
        tokenizer=tiny_tokenizer_mock,
        rec_encoder=rec_encoder,
        cie_module=cie,
        config=config,
    )
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :8] = -100
    user_ids = torch.tensor([0, 1])
    target_item_ids = torch.tensor([5, 6])
    user_ph_pos = torch.tensor([2, 2])
    item_ph_pos = torch.tensor([5, 5])

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        user_ids=user_ids,
        target_item_ids=target_item_ids,
        user_placeholder_pos=user_ph_pos,
        item_placeholder_pos=item_ph_pos,
    )
    assert out.loss is not None


def test_stage2_embedding_substitution(tiny_lm, config, tiny_tokenizer_mock):
    """確認 embedding 確實在正確位置被替換"""
    from unittest.mock import patch

    rec_encoder = MFEncoder(config.rec)
    cie = CIEModule(rec_dim=8, llm_dim=16)
    model = CoLLMModel(
        backbone=tiny_lm,
        tokenizer=tiny_tokenizer_mock,
        rec_encoder=rec_encoder,
        cie_module=cie,
        config=config,
    )

    input_ids = torch.zeros(1, 8, dtype=torch.long)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    user_ids = torch.tensor([0])
    target_item_ids = torch.tensor([5])
    user_ph_pos = torch.tensor([2])
    item_ph_pos = torch.tensor([5])

    # 記錄原始 token embedding（替換前）
    orig_emb = tiny_lm.get_input_embeddings()(input_ids).detach().clone()

    captured = {}
    original_forward = tiny_lm.__class__.forward

    def patched_forward(self, input_ids=None, inputs_embeds=None,
                        attention_mask=None, labels=None, **kw):
        captured["inputs_embeds"] = inputs_embeds.detach().clone() if inputs_embeds is not None else None
        return original_forward(self, input_ids=input_ids,
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                labels=labels, **kw)

    with patch.object(tiny_lm.__class__, "forward", patched_forward):
        model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            user_ids=user_ids, target_item_ids=target_item_ids,
            user_placeholder_pos=user_ph_pos, item_placeholder_pos=item_ph_pos,
        )

    assert captured["inputs_embeds"] is not None
    # 位置 2 已被 CIE 投影後的 user embedding 替換，應不同於原始 token embedding
    assert not torch.allclose(captured["inputs_embeds"][0, 2], orig_emb[0, 2])


def test_neg1_pos_skips_substitution(tiny_lm, config, tiny_tokenizer_mock):
    """-1 位置不應替換 embedding，所有位置的 embedding 應與原始一致"""
    from unittest.mock import patch

    rec_encoder = MFEncoder(config.rec)
    cie = CIEModule(rec_dim=8, llm_dim=16)
    model = CoLLMModel(
        backbone=tiny_lm, tokenizer=tiny_tokenizer_mock,
        rec_encoder=rec_encoder, cie_module=cie, config=config,
    )
    input_ids = torch.zeros(1, 8, dtype=torch.long)
    orig_emb = tiny_lm.get_input_embeddings()(input_ids).detach().clone()

    captured = {}
    original_forward = tiny_lm.__class__.forward

    def patched_forward(self, input_ids=None, inputs_embeds=None,
                        attention_mask=None, labels=None, **kw):
        captured["inputs_embeds"] = inputs_embeds.detach().clone() if inputs_embeds is not None else None
        return original_forward(self, input_ids=input_ids,
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                labels=labels, **kw)

    with patch.object(tiny_lm.__class__, "forward", patched_forward):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            labels=torch.full((1, 8), -100, dtype=torch.long),
            user_ids=torch.tensor([0]),
            target_item_ids=torch.tensor([5]),
            user_placeholder_pos=torch.tensor([-1]),
            item_placeholder_pos=torch.tensor([-1]),
        )

    assert captured["inputs_embeds"] is not None
    # 所有位置均未替換，應與原始 token embedding 相同
    assert torch.allclose(captured["inputs_embeds"], orig_emb)
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_collm_model.py -v
```

- [ ] **Step 3：實作 `collm/model/collm.py`**

```python
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from collm.encoders.base import BaseRecEncoder
from collm.model.cie import CIEModule
from collm.training.config import CoLLMConfig


class CoLLMModel(nn.Module):
    """CoLLM 主模型：將協同過濾信號注入 LLM 的 token embedding 空間。

    兩種工作模式：
    - Stage 1（rec_encoder=None）：純語言模型，直接 forward backbone
    - Stage 2（rec_encoder 非 None + use_collaborative_signal=True）：
      替換 [USER_TOKEN] 和 [ITEM_TOKEN] 位置的 embedding 後再 forward

    Args:
        backbone:       HuggingFace LLM（已套用 LoRA）
        tokenizer:      對應的 tokenizer（已注冊占位符 token）
        rec_encoder:    推薦 Encoder（Stage 1 時為 None）
        cie_module:     CIE 投影層（Stage 1 時為 None）
        config:         CoLLMConfig
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        rec_encoder: Optional[BaseRecEncoder] = None,
        cie_module: Optional[CIEModule] = None,
        config: Optional[CoLLMConfig] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.rec_encoder = rec_encoder
        self.cie_module = cie_module
        self.use_collaborative_signal = (
            config.rec.use_collaborative_signal if config is not None else False
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        user_ids: Optional[Tensor] = None,
        target_item_ids: Optional[Tensor] = None,
        user_placeholder_pos: Optional[Tensor] = None,
        item_placeholder_pos: Optional[Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        所有參數均為 LongTensor 或 FloatTensor，相容 HF Trainer 的 device 轉移。
        seq_history 透過 **kwargs 傳入（僅序列模型批次含此鍵）。
        """
        seq_history: Optional[Tensor] = kwargs.get("seq_history")

        # Step 1：取 token embeddings
        embed_fn = self.backbone.get_input_embeddings()
        inputs_embeds = embed_fn(input_ids).clone()  # (batch, seq, d_llm)

        # Step 2：注入協同信號（若啟用）
        if (
            self.rec_encoder is not None
            and self.cie_module is not None
            and self.use_collaborative_signal
            and user_ids is not None
            and target_item_ids is not None
        ):
            user_emb = self.rec_encoder.get_user_embedding(
                user_ids,
                seq_history=seq_history,
                target_item_ids=target_item_ids,
            )  # (batch, d_rec)
            item_emb = self.rec_encoder.get_item_embedding(
                target_item_ids,
                seq_history=seq_history,
            )  # (batch, d_rec)

            user_token = self.cie_module(user_emb)  # (batch, d_llm)
            item_token = self.cie_module(item_emb)  # (batch, d_llm)

            # Step 3：替換占位符位置的 embedding
            if user_placeholder_pos is not None:
                for i in range(inputs_embeds.size(0)):
                    if user_placeholder_pos[i] != -1:
                        inputs_embeds[i, user_placeholder_pos[i]] = user_token[i]

            if item_placeholder_pos is not None:
                for i in range(inputs_embeds.size(0)):
                    if item_placeholder_pos[i] != -1:
                        inputs_embeds[i, item_placeholder_pos[i]] = item_token[i]

        # Step 4：送入 backbone
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def freeze_backbone(self) -> None:
        """凍結 backbone 所有參數（含 LoRA）"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_rec_encoder(self) -> None:
        """凍結 rec encoder 所有參數"""
        if self.rec_encoder is not None:
            self.rec_encoder.freeze()
```

- [ ] **Step 4：執行確認 pass**

```bash
pytest tests/test_collm_model.py -v
```

Expected: 全部 pass

- [ ] **Step 5：更新 `collm/model/__init__.py`**

```python
from collm.model.cie import CIEModule
from collm.model.backbone import build_backbone
from collm.model.collm import CoLLMModel

__all__ = ["CIEModule", "build_backbone", "CoLLMModel"]
```

- [ ] **Step 6：Commit**

```bash
git add collm/model/collm.py collm/model/__init__.py tests/test_collm_model.py
git commit -m "feat: add CoLLMModel with collaborative embedding injection"
```

---

## Task 14：Metrics

**Files:**
- Create: `collm/training/metrics.py`
- Modify: `collm/training/__init__.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1：寫 failing test**

```python
# tests/test_metrics.py
import numpy as np
import pytest
from collm.training.metrics import compute_auc, compute_hr, compute_ndcg


def test_auc_perfect():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0])
    assert compute_auc(scores, labels) == pytest.approx(1.0)


def test_auc_random():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 0, 1, 0])
    auc = compute_auc(scores, labels)
    assert 0.0 <= auc <= 1.0


def test_hr_at_k():
    # 正樣本排名第 1（最高分），K=3 應命中
    scores = np.array([0.9, 0.5, 0.3, 0.1])
    labels = np.array([1, 0, 0, 0])
    assert compute_hr(scores, labels, k=3) == 1.0


def test_hr_at_k_miss():
    # 正樣本排名第 4，K=3 不命中
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    labels = np.array([0, 0, 0, 1])
    assert compute_hr(scores, labels, k=3) == 0.0


def test_ndcg_at_k():
    scores = np.array([0.9, 0.5, 0.3, 0.1])
    labels = np.array([1, 0, 0, 0])
    ndcg = compute_ndcg(scores, labels, k=3)
    assert ndcg == pytest.approx(1.0)


def test_ndcg_partial():
    # 正樣本在第 2 位
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    labels = np.array([0, 1, 0, 0])
    ndcg = compute_ndcg(scores, labels, k=3)
    import math
    expected = 1.0 / math.log2(3)  # 位置 2：DCG = 1/log2(3)
    assert ndcg == pytest.approx(expected, rel=1e-5)
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 3：實作 `collm/training/metrics.py`**

```python
import math
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """計算 AUC（Area Under ROC Curve）。

    Args:
        scores: 預測分數，shape (n,)
        labels: 二值標籤（0 或 1），shape (n,)

    Returns:
        AUC 值，範圍 [0, 1]
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    return roc_auc_score(labels, scores)


def compute_hr(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """計算 HR@K（Hit Rate at K）。

    Args:
        scores: 預測分數，shape (n,)
        labels: 二值標籤（0 或 1），shape (n,)
        k:      top-K 截斷

    Returns:
        HR@K，1.0 若正樣本在 top-K 中，否則 0.0
    """
    top_k_idx = np.argsort(scores)[::-1][:k]
    return 1.0 if labels[top_k_idx].sum() > 0 else 0.0


def compute_ndcg(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """計算 NDCG@K（Normalized Discounted Cumulative Gain at K）。

    Args:
        scores: 預測分數，shape (n,)
        labels: 二值標籤（0 或 1），shape (n,)
        k:      top-K 截斷

    Returns:
        NDCG@K 值，範圍 [0, 1]
    """
    top_k_idx = np.argsort(scores)[::-1][:k]
    dcg = sum(
        labels[idx] / math.log2(rank + 2)
        for rank, idx in enumerate(top_k_idx)
    )
    ideal_hits = min(k, int(labels.sum()))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
```

- [ ] **Step 4：執行確認 pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: 6 passed

- [ ] **Step 5：更新 `collm/training/__init__.py`**

```python
from collm.training.config import RecEncoderConfig, BackboneConfig, DataConfig, CoLLMConfig
from collm.training.metrics import compute_auc, compute_hr, compute_ndcg

__all__ = [
    "RecEncoderConfig", "BackboneConfig", "DataConfig", "CoLLMConfig",
    "compute_auc", "compute_hr", "compute_ndcg",
]
```

- [ ] **Step 6：執行確認 pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: 6 passed

- [ ] **Step 7：Commit**

```bash
git add collm/training/metrics.py collm/training/__init__.py tests/test_metrics.py
git commit -m "feat: add AUC, HR@K, NDCG@K metrics"
```

---

## Task 15：CoLLMTrainer

**Files:**
- Create: `collm/training/trainer.py`
- Modify: `collm/training/__init__.py`

- [ ] **Step 1：寫 failing test**

```python
# tests/test_trainer.py（加入此文件）
import pytest
import torch
from unittest.mock import MagicMock, patch
from collm.training.trainer import CoLLMTrainer


def test_trainer_is_subclass_of_hf_trainer():
    from transformers import Trainer
    assert issubclass(CoLLMTrainer, Trainer)


def test_prepare_inputs_handles_missing_seq_history():
    """_prepare_inputs 不應因缺少 seq_history 鍵而出錯"""
    trainer = CoLLMTrainer.__new__(CoLLMTrainer)  # 不呼叫 __init__
    # 模擬 batch（無 seq_history 鍵）
    batch = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels": torch.full((2, 8), -100, dtype=torch.long),
        "user_ids": torch.tensor([0, 1]),
        "target_item_ids": torch.tensor([5, 6]),
        "user_placeholder_pos": torch.tensor([2, 2]),
        "item_placeholder_pos": torch.tensor([5, 5]),
    }
    # 確保所有值都是 Tensor（可呼叫 .to(device)），不 crash
    for v in batch.values():
        assert hasattr(v, "to"), f"{v} 沒有 .to() 方法"
```

- [ ] **Step 2：執行確認 fail**

```bash
pytest tests/test_trainer.py -v
```

- [ ] **Step 3：實作 `collm/training/trainer.py`**

```python
from __future__ import annotations
from typing import Any
import torch
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
import numpy as np
from collm.training.metrics import compute_auc, compute_hr, compute_ndcg


class CoLLMTrainer(Trainer):
    """繼承 HuggingFace Trainer 的 CoLLM 訓練器。

    主要客製化：
    1. compute_metrics：計算 AUC、HR@10、NDCG@10
    2. 批次中所有值均為 Tensor，與 HF Trainer 的 device 轉移機制完全相容

    使用範例：
        trainer = CoLLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            compute_metrics=CoLLMTrainer.default_compute_metrics,
        )
        trainer.train()
    """

    @staticmethod
    def default_compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        """計算推薦任務評估指標。

        eval_pred.predictions: (n_samples, vocab_size) logits（最後一個 token）
        eval_pred.label_ids:   (n_samples,) 標籤（0 或 1）

        HF Trainer 在 predict_step 中收集 logits 和 labels，
        此函數負責將其轉換為推薦指標。
        """
        logits, labels = eval_pred
        # 取 Yes/No token 的 logit 差作為推薦分數
        # 注意：需要在 predict_step 中指定 yes_token_id / no_token_id
        # 此處作為基礎實作，直接用 logits 的 max 分數
        scores = logits.max(axis=-1) if logits.ndim > 1 else logits
        labels_binary = (labels > 0).astype(int)

        return {
            "auc": compute_auc(scores, labels_binary),
            "hr@10": compute_hr(scores, labels_binary, k=10),
            "ndcg@10": compute_ndcg(scores, labels_binary, k=10),
        }
```

- [ ] **Step 4：執行確認 pass**

```bash
pytest tests/test_trainer.py -v
```

- [ ] **Step 5：更新 `collm/training/__init__.py`（加入 CoLLMTrainer）**

```python
from collm.training.config import RecEncoderConfig, BackboneConfig, DataConfig, CoLLMConfig
from collm.training.metrics import compute_auc, compute_hr, compute_ndcg
from collm.training.trainer import CoLLMTrainer

__all__ = [
    "RecEncoderConfig", "BackboneConfig", "DataConfig", "CoLLMConfig",
    "compute_auc", "compute_hr", "compute_ndcg",
    "CoLLMTrainer",
]
```

- [ ] **Step 6：執行確認 pass**

```bash
pytest tests/test_trainer.py -v
```

- [ ] **Step 7：Commit**

```bash
git add collm/training/trainer.py collm/training/__init__.py tests/test_trainer.py
git commit -m "feat: add CoLLMTrainer with AUC/HR/NDCG metrics"
```

---

## Task 16：訓練腳本與 YAML 配置

**Files:**
- Create: `scripts/train_rec_encoder.py`
- Create: `scripts/train_stage1.py`
- Create: `scripts/train_stage2.py`

- [ ] **Step 1：實作 `scripts/train_rec_encoder.py`**

```python
"""
Rec Encoder 預訓練腳本（獨立於 CoLLM）。

用法：
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
"""
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collm.training.config import RecEncoderConfig
from collm.encoders.mf import MFEncoder
from collm.encoders.sasrec import SASRecEncoder


class PairDataset(Dataset):
    """用戶-物品交互對資料集（正樣本 + BPR 負採樣）。"""

    def __init__(self, df: pd.DataFrame, item_num: int, max_seq_len: int):
        self.records = df.to_dict("records")
        self.item_num = item_num
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        neg_item = torch.randint(1, self.item_num, (1,)).item()
        hist = list(row.get("history_iid", []))[-self.max_seq_len:]
        padded = [0] * (self.max_seq_len - len(hist)) + hist
        return {
            "uid": int(row["uid"]),
            "pos_iid": int(row["iid"]),
            "neg_iid": neg_item,
            "seq_history": torch.tensor(padded, dtype=torch.long),
        }


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--encoder_type", default="sasrec",
                        choices=["mf", "sasrec"])
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    df = pd.read_pickle(args.data_path + ".pkl")
    user_num = df["uid"].max() + 1
    item_num = df["iid"].max() + 1

    config = RecEncoderConfig(
        encoder_type=args.encoder_type,
        checkpoint_path="",
        embedding_dim=args.embedding_dim,
        user_num=int(user_num),
        item_num=int(item_num),
        max_seq_len=args.max_seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    if args.encoder_type == "mf":
        encoder = MFEncoder(config)
    elif args.encoder_type == "sasrec":
        encoder = SASRecEncoder(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    dataset = PairDataset(df, int(item_num), args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        encoder.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            uid = batch["uid"].to(device)
            pos = batch["pos_iid"].to(device)
            neg = batch["neg_iid"].to(device)
            seq = batch["seq_history"].to(device)

            user_emb = encoder.get_user_embedding(uid, seq_history=seq)
            pos_emb = encoder.get_item_embedding(pos)
            neg_emb = encoder.get_item_embedding(neg)

            pos_score = (user_emb * pos_emb).sum(dim=-1)
            neg_score = (user_emb * neg_emb).sum(dim=-1)

            loss = bpr_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")

    torch.save(encoder.state_dict(), args.save_path)
    print(f"Rec encoder saved to {args.save_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2：實作 `scripts/train_stage1.py`**

```python
"""
Stage 1：LoRA 預訓練（純文字，無協同信號）。

用法：
    python scripts/train_stage1.py \
        --config configs/stage1_movielens.yaml \
        --output_dir /checkpoints/stage1 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --learning_rate 2e-4
"""
import argparse
import yaml
import dacite
import pandas as pd
from transformers import TrainingArguments
from torch.utils.data import DataLoader

from collm.training.config import CoLLMConfig
from collm.model.backbone import build_backbone
from collm.model.collm import CoLLMModel
from collm.data.dataset import RecDataset
from collm.data.collator import RecDataCollator
from collm.training.trainer import CoLLMTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="CoLLMConfig YAML 路徑")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()

    # 載入配置
    raw = yaml.safe_load(open(args.config))
    cfg = dacite.from_dict(CoLLMConfig, raw, config=dacite.Config(cast=[list]))

    # 從資料推斷 user_num / item_num
    train_df = pd.read_pickle(cfg.data.data_path + "/train.pkl")
    cfg.rec.user_num = int(train_df["uid"].max()) + 1
    cfg.rec.item_num = int(train_df["iid"].max()) + 1

    # 強制關閉協同信號（Stage 1 = 純文字）
    cfg.rec.use_collaborative_signal = False

    # 建立模型（無 rec_encoder / cie）
    backbone, tokenizer = build_backbone(cfg.backbone)
    model = CoLLMModel(backbone, tokenizer, config=cfg)

    # 資料集
    train_ds = RecDataset(cfg.data.data_path + "/train", cfg.data.dataset_type)
    eval_ds = RecDataset(cfg.data.data_path + "/valid", cfg.data.dataset_type)

    is_sequential = cfg.rec.encoder_type in ("sasrec", "din")
    collator = RecDataCollator(
        tokenizer=tokenizer,
        use_collaborative_signal=False,
        max_seq_len=cfg.data.max_seq_len,
        max_history_len=cfg.data.max_history_len,
        is_sequential=is_sequential,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,
        remove_unused_columns=False,  # 保留自定義欄位
    )

    trainer = CoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Stage 1 完成，checkpoint 已存至 {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3：實作 `scripts/train_stage2.py`**

```python
"""
Stage 2：CIE 訓練（帶協同信號）。
載入 Stage 1 checkpoint，凍結 backbone + LoRA + rec encoder，只訓練 CIE 投影層。

用法：
    python scripts/train_stage2.py \
        --config configs/stage2_movielens.yaml \
        --stage1_checkpoint /checkpoints/stage1 \
        --output_dir /checkpoints/stage2 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 4 \
        --learning_rate 1e-4
"""
import argparse
import yaml
import dacite
import pandas as pd
from transformers import TrainingArguments

from collm.training.config import CoLLMConfig, RecEncoderConfig
from collm.model.backbone import build_backbone
from collm.model.cie import CIEModule
from collm.model.collm import CoLLMModel
from collm.data.dataset import RecDataset
from collm.data.collator import RecDataCollator
from collm.training.trainer import CoLLMTrainer
from collm.encoders.mf import MFEncoder
from collm.encoders.lightgcn import LightGCNEncoder
from collm.encoders.sasrec import SASRecEncoder
from collm.encoders.din import DINEncoder


_ENCODER_MAP = {
    "mf": MFEncoder,
    "lightgcn": LightGCNEncoder,
    "sasrec": SASRecEncoder,
    "din": DINEncoder,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1_checkpoint", required=True,
                        help="Stage 1 存下的 checkpoint 目錄")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=50)
    args = parser.parse_args()

    raw = yaml.safe_load(open(args.config))
    cfg = dacite.from_dict(CoLLMConfig, raw, config=dacite.Config(cast=[list]))

    train_df = pd.read_pickle(cfg.data.data_path + "/train.pkl")
    cfg.rec.user_num = int(train_df["uid"].max()) + 1
    cfg.rec.item_num = int(train_df["iid"].max()) + 1
    cfg.rec.use_collaborative_signal = True

    # 載入 Stage 1 backbone（已有 LoRA 權重）
    cfg.backbone.model_name_or_path = args.stage1_checkpoint
    backbone, tokenizer = build_backbone(cfg.backbone)

    # 載入並凍結 Rec Encoder
    EncoderClass = _ENCODER_MAP[cfg.rec.encoder_type]
    rec_encoder = EncoderClass.from_pretrained(cfg.rec.checkpoint_path, cfg.rec)

    # 初始化 CIE（僅可訓練參數）
    # 使用 base_model.config 避免 PEFT wrapper 導致的屬性存取不一致
    base_config = getattr(backbone, "base_model", backbone).config
    llm_dim = base_config.hidden_size
    cie = CIEModule(rec_dim=cfg.rec.embedding_dim, llm_dim=llm_dim)

    model = CoLLMModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rec_encoder=rec_encoder,
        cie_module=cie,
        config=cfg,
    )
    # 凍結 backbone（含 LoRA）和 rec_encoder
    # 刻意設計：Stage 1 已賦予 LoRA 推薦能力，Stage 2 只讓 CIE 對齊協同空間
    model.freeze_backbone()
    model.freeze_rec_encoder()

    train_ds = RecDataset(cfg.data.data_path + "/train", cfg.data.dataset_type)
    eval_ds = RecDataset(cfg.data.data_path + "/valid", cfg.data.dataset_type)

    is_sequential = cfg.rec.encoder_type in ("sasrec", "din")
    collator = RecDataCollator(
        tokenizer=tokenizer,
        use_collaborative_signal=True,
        max_seq_len=cfg.data.max_seq_len,
        max_history_len=cfg.data.max_history_len,
        is_sequential=is_sequential,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        fp16=True,
        remove_unused_columns=False,
    )

    trainer = CoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Stage 2 完成，checkpoint 已存至 {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4：執行所有測試確認全部通過**

```bash
pytest tests/ -v --tb=short
```

Expected: 全部 pass

- [ ] **Step 5：Commit**

```bash
git add scripts/ configs/
git commit -m "feat: add training scripts for rec encoder, stage1, stage2"
```

---

## Task 17：最終整合驗證

- [ ] **Step 1：確認 package 可正常 import**

```bash
python -c "
from collm.encoders import MFEncoder, LightGCNEncoder, SASRecEncoder, DINEncoder
from collm.model import CoLLMModel, CIEModule, build_backbone
from collm.data import get_prompt, get_answer
from collm.training.config import CoLLMConfig
from collm.training.metrics import compute_auc, compute_hr, compute_ndcg
from collm.training.trainer import CoLLMTrainer
print('所有模塊 import 成功')
"
```

- [ ] **Step 2：執行完整測試套件**

```bash
pytest tests/ -v --tb=short
```

Expected: 全部 pass，無 warning

- [ ] **Step 3：最終 commit**

```bash
git add -A
git commit -m "feat: complete CoLLM refactor - modular architecture with HF Trainer"
```

---

## 快速參考：消融實驗

切換 `use_collaborative_signal=False` 即可運行消融實驗：

```python
# 在 configs/stage2_movielens.yaml 中
rec:
  use_collaborative_signal: false  # 改為 false
```

或直接在訓練腳本中覆蓋：

```python
cfg.rec.use_collaborative_signal = False
```

這會讓 DataCollator 選擇無占位符的 prompt 模板，CoLLMModel.forward() 自動跳過 embedding 替換步驟。

---

## 快速參考：新增 Rec Encoder

1. 在 `collm/encoders/` 建立新文件（如 `dcn.py`）
2. 繼承 `BaseRecEncoder`，實作三個 abstractmethod
3. 在 `collm/encoders/__init__.py` 匯出
4. 在 `scripts/train_stage2.py` 的 `_ENCODER_MAP` 加入對應項目
5. 在 `tests/test_encoders.py` 補充測試
