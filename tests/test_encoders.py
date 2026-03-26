import pytest
import torch
from collm.encoders.base import BaseRecEncoder


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
