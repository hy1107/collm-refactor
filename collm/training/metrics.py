import math
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """計算 AUC（Area Under ROC Curve）。

    Args:
        scores: 預測分數，shape (n,)
        labels: 二值標籤（0 或 1），shape (n,)

    Returns:
        AUC 值，範圍 [0, 1]
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    return roc_auc_score(labels, scores)


def compute_hr(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """計算 HR@K（Hit Rate at K）。

    Args:
        scores: 預測分數，shape (n,)
        labels: 二值標籤（0 或 1），shape (n,)
        k:      top-K 截斷

    Returns:
        HR@K，1.0 若正樣本在 top-K 中，否則 0.0
    """
    top_k_idx = np.argsort(scores)[::-1][:k]
    return 1.0 if labels[top_k_idx].sum() > 0 else 0.0


def compute_ndcg(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """計算 NDCG@K（Normalized Discounted Cumulative Gain at K）。

    Args:
        scores: 預測分數，shape (n,)
        labels: 二值標籤（0 或 1），shape (n,)
        k:      top-K 截斷

    Returns:
        NDCG@K 值，範圍 [0, 1]
    """
    top_k_idx = np.argsort(scores)[::-1][:k]
    dcg = sum(
        labels[idx] / math.log2(rank + 2)
        for rank, idx in enumerate(top_k_idx)
    )
    ideal_hits = min(k, int(labels.sum()))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
