from __future__ import annotations
from typing import Any
import torch
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
import numpy as np
from collm.training.metrics import compute_auc, compute_hr, compute_ndcg


class CoLLMTrainer(Trainer):
    """繼承 HuggingFace Trainer 的 CoLLM 訓練器。

    主要客製化：
    1. compute_metrics：計算 AUC、HR@10、NDCG@10
    2. 批次中所有值均為 Tensor，與 HF Trainer 的 device 轉移機制完全相容

    使用範例：
        trainer = CoLLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            compute_metrics=CoLLMTrainer.default_compute_metrics,
        )
        trainer.train()
    """

    @staticmethod
    def default_compute_metrics(eval_pred: EvalPrediction) -> dict:
        """計算推薦任務評估指標。

        eval_pred.predictions: (n_samples, vocab_size) logits（最後一個 token）
        eval_pred.label_ids:   (n_samples,) 標籤（0 或 1）
        """
        logits, labels = eval_pred
        scores = logits.max(axis=-1) if logits.ndim > 1 else logits
        labels_binary = (labels > 0).astype(int)

        return {
            "auc": compute_auc(scores, labels_binary),
            "hr@10": compute_hr(scores, labels_binary, k=10),
            "ndcg@10": compute_ndcg(scores, labels_binary, k=10),
        }
