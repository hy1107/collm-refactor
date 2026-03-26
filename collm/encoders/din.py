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
        seq_history = kwargs.get("seq_history")
        target_item_ids = kwargs.get("target_item_ids")

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
