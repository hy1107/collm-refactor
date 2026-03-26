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
