import pytest
import torch
import scipy.sparse as sp
import numpy as np
from collm.encoders.base import BaseRecEncoder
from collm.encoders.mf import MFEncoder
from collm.encoders.lightgcn import LightGCNEncoder
from collm.encoders.sasrec import SASRecEncoder
from collm.encoders.din import DINEncoder


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


def _make_adj_matrix(user_num, item_num):
    """建立一個簡單的 user-item 交互鄰接矩陣（COO 格式 → sparse tensor）"""
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
