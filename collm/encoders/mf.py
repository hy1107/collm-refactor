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
