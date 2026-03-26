import numpy as np
import pytest
from collm.training.metrics import compute_auc, compute_hr, compute_ndcg


def test_auc_perfect():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0])
    assert compute_auc(scores, labels) == pytest.approx(1.0)


def test_auc_random():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 0, 1, 0])
    auc = compute_auc(scores, labels)
    assert 0.0 <= auc <= 1.0


def test_hr_at_k():
    # 正樣本排名第 1（最高分），K=3 應命中
    scores = np.array([0.9, 0.5, 0.3, 0.1])
    labels = np.array([1, 0, 0, 0])
    assert compute_hr(scores, labels, k=3) == 1.0


def test_hr_at_k_miss():
    # 正樣本排名第 4，K=3 不命中
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    labels = np.array([0, 0, 0, 1])
    assert compute_hr(scores, labels, k=3) == 0.0


def test_ndcg_at_k():
    scores = np.array([0.9, 0.5, 0.3, 0.1])
    labels = np.array([1, 0, 0, 0])
    ndcg = compute_ndcg(scores, labels, k=3)
    assert ndcg == pytest.approx(1.0)


def test_ndcg_partial():
    # 正樣本在第 2 位
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    labels = np.array([0, 1, 0, 0])
    ndcg = compute_ndcg(scores, labels, k=3)
    import math
    expected = 1.0 / math.log2(3)  # 位置 2：DCG = 1/log2(3)
    assert ndcg == pytest.approx(expected, rel=1e-5)
