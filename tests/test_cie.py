import torch
from collm.model.cie import CIEModule


def test_cie_output_shape():
    rec_dim, llm_dim = 64, 4096
    cie = CIEModule(rec_dim, llm_dim)
    x = torch.randn(4, rec_dim)
    out = cie(x)
    assert out.shape == (4, llm_dim)


def test_cie_small_dims():
    cie = CIEModule(8, 16)
    x = torch.randn(3, 8)
    out = cie(x)
    assert out.shape == (3, 16)


def test_cie_is_differentiable():
    cie = CIEModule(8, 16)
    x = torch.randn(2, 8, requires_grad=True)
    out = cie(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
