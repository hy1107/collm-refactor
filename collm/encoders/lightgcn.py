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

    def _propagate(self) -> tuple:
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
