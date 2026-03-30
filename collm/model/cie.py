import torch.nn as nn
from torch import Tensor


class CIEModule(nn.Module):
    """CIE（Collaborative Information Embedding）投影模塊。

    將協同過濾 encoder 輸出的 embedding（維度 d_rec）透過兩層 MLP 投影到
    LLM 的 token embedding 空間（維度 d_llm），產生可直接注入 LLM 的虛擬 token。

    對應原始論文實作：d_rec → d_rec * proj_mid_times → d_llm，使用 ReLU。

    Args:
        rec_dim:         協同 embedding 維度（d_rec）
        llm_dim:         LLM token embedding 維度（d_llm，通常為 4096）
        proj_mid_times:  hidden dim 倍數（hidden = d_rec * proj_mid_times），預設 10
    """

    def __init__(self, rec_dim: int, llm_dim: int, proj_mid_times: int = 10):
        super().__init__()
        hidden = rec_dim * proj_mid_times
        self.proj = nn.Sequential(
            nn.Linear(rec_dim, hidden),
            nn.ReLU(),
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
