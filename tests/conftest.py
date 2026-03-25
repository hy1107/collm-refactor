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
