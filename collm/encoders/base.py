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
