from __future__ import annotations
from typing import Callable
import numpy as np
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
from collm.training.metrics import compute_auc, compute_hr, compute_ndcg


class CoLLMTrainer(Trainer):
    """繼承 HuggingFace Trainer 的 CoLLM 訓練器。

    主要客製化：
    1. compute_metrics：計算 AUC、HR@10、NDCG@10
    2. 批次中所有值均為 Tensor，與 HF Trainer 的 device 轉移機制完全相容

    使用範例：
        yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
        no_id  = tokenizer.encode(" No",  add_special_tokens=False)[0]
        trainer = CoLLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            compute_metrics=CoLLMTrainer.make_compute_metrics(yes_id, no_id),
        )
        trainer.train()
    """

    @staticmethod
    def make_compute_metrics(yes_token_id: int, no_token_id: int) -> Callable:
        """建立 compute_metrics 函式，以 Yes/No token logit 差值作為推薦分數。

        Args:
            yes_token_id: tokenizer 中 " Yes" 對應的 token ID
            no_token_id:  tokenizer 中 " No"  對應的 token ID

        eval_pred.predictions: (batch, seq_len, vocab_size) 或 (batch, vocab_size)
        eval_pred.label_ids:   (batch, seq_len)，prompt 部分為 -100，答案位置為實際 token ID
        """
        def _compute_metrics(eval_pred: EvalPrediction) -> dict:
            logits, labels = eval_pred
            scores, labels_binary = [], []

            for i in range(len(logits)):
                if logits.ndim == 3:
                    # 找到第一個非 -100 的 label 位置（即答案 token 位置）
                    valid = np.where(labels[i] != -100)[0]
                    if len(valid) == 0:
                        continue
                    pos = valid[0]
                    score = float(logits[i, pos, yes_token_id] - logits[i, pos, no_token_id])
                    label = 1 if int(labels[i, pos]) == yes_token_id else 0
                else:
                    # logits 已經是 (batch, vocab_size)
                    score = float(logits[i, yes_token_id] - logits[i, no_token_id])
                    label = int(labels[i]) if labels.ndim == 1 else int(labels[i, 0])
                    label = 1 if label == yes_token_id else 0

                scores.append(score)
                labels_binary.append(label)

            scores = np.array(scores, dtype=np.float32)
            labels_binary = np.array(labels_binary, dtype=np.int32)

            return {
                "auc": compute_auc(scores, labels_binary),
                "hr@10": compute_hr(scores, labels_binary, k=10),
                "ndcg@10": compute_ndcg(scores, labels_binary, k=10),
            }

        return _compute_metrics
