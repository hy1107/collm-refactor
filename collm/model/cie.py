import torch.nn as nn
from torch import Tensor


class CIEModule(nn.Module):
    """CIE（Collaborative Information Embedding）投影模塊。

    將協同過濾 encoder 輸出的 embedding（維度 d_rec）透過兩層 MLP 投影到
    LLM 的 token embedding 空間（維度 d_llm），產生可直接注入 LLM 的虛擬 token。

    對應原始論文的多層映射網路設計（rec_dim → hidden → llm_dim，hidden = 均值）。

    Args:
        rec_dim:  協同 embedding 維度（d_rec）
        llm_dim:  LLM token embedding 維度（d_llm，通常為 4096）
    """

    def __init__(self, rec_dim: int, llm_dim: int):
        super().__init__()
        hidden = (rec_dim + llm_dim) // 2
        self.proj = nn.Sequential(
            nn.Linear(rec_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, rec_embedding: Tensor) -> Tensor:
        """
        參數:
            rec_embedding: FloatTensor, shape (batch, d_rec)
        返回:
            FloatTensor, shape (batch, d_llm)
        """
        return self.proj(rec_embedding)
