import pytest
import torch
from collm.encoders.base import BaseRecEncoder
from collm.encoders.mf import MFEncoder


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
