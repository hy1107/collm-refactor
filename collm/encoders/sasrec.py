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

    def forward(self, x: Tensor, key_padding_mask=None) -> Tensor:
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
