from __future__ import annotations
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
from collm.training.metrics import compute_auc, compute_hr, compute_ndcg


class CoLLMTrainer(Trainer):
    """繼承 HuggingFace Trainer 的 CoLLM 訓練器。

    prediction_step 只保留每樣本的 Yes/No logit，避免收集全 vocab logits 造成 OOM。
    compute_metrics 接收 (N, 2) 的 scores 陣列（[:, 0]=yes, [:, 1]=no）。
    """

    def __init__(self, *args, yes_token_id: int, no_token_id: int, **kwargs):
        # 移除 compute_metrics，改由內部處理
        kwargs.pop("compute_metrics", None)
        super().__init__(*args, compute_metrics=self._make_metrics(), **kwargs)
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

    def _make_metrics(self):
        def _compute(eval_pred: EvalPrediction) -> dict:
            scores_2d, labels_1d = eval_pred.predictions, eval_pred.label_ids
            # scores_2d: (N, 2)  labels_1d: (N,)
            scores = (scores_2d[:, 0] - scores_2d[:, 1]).astype(np.float32)
            labels = labels_1d.astype(np.int32)
            # 過濾 NaN/inf（fp16 溢出）及 padding 殘留（label < 0）
            nan_mask = ~np.isfinite(scores)
            pad_mask = labels < 0
            print(f"[eval] 總樣本: {len(scores)}, NaN/inf: {nan_mask.sum()} ({nan_mask.mean():.2%}), padding: {pad_mask.sum()}")
            valid = ~nan_mask & ~pad_mask
            scores, labels = scores[valid], labels[valid]
            return {
                "auc":     compute_auc(scores, labels),
                "hr@10":   compute_hr(scores, labels, k=10),
                "ndcg@10": compute_ndcg(scores, labels, k=10),
            }
        return _compute

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """BCE loss on the Yes-token logit only.

        論文 Section IV-B Eq.(5) 明確使用 binary cross-entropy loss：
        只取 answer token 位置的 Yes-token logit，對二元標籤（1=Yes, 0=No）計算 BCEWithLogits。
        不使用 HuggingFace 預設的 causal LM cross-entropy。
        """
        outputs = model(**inputs)
        seq_labels = inputs.get("labels")   # (batch, seq_len), -100 everywhere except answer pos
        logits = outputs.logits              # (batch, seq_len, vocab)

        batch_size = logits.shape[0]
        yes_logits = logits.new_zeros(batch_size)
        binary_labels = logits.new_zeros(batch_size)

        for i in range(batch_size):
            valid = (seq_labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                continue
            pos = valid[0]
            yes_logits[i] = logits[i, pos, self.yes_token_id]
            binary_labels[i] = 1.0 if int(seq_labels[i, pos]) == self.yes_token_id else 0.0

        loss = F.binary_cross_entropy_with_logits(yes_logits, binary_labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """只保留 Yes/No logit，避免 OOM。回傳 (loss, scores(N,2), binary_labels(N,))。"""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss if hasattr(outputs, "loss") else None
        if prediction_loss_only:
            return (loss, None, None)

        logits = outputs.logits          # (batch, seq_len, vocab)
        seq_labels = inputs.get("labels")  # (batch, seq_len)
        batch_size = logits.shape[0]

        yes_scores = torch.zeros(batch_size, device=logits.device, dtype=torch.float32)
        no_scores  = torch.zeros(batch_size, device=logits.device, dtype=torch.float32)
        binary_labels = torch.zeros(batch_size, device=logits.device, dtype=torch.long)

        for i in range(batch_size):
            valid = (seq_labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                continue
            pos = valid[0]
            yes_scores[i] = logits[i, pos, self.yes_token_id].float()
            no_scores[i]  = logits[i, pos, self.no_token_id].float()
            binary_labels[i] = 1 if int(seq_labels[i, pos]) == self.yes_token_id else 0

        scores = torch.stack([yes_scores, no_scores], dim=1)  # (batch, 2)
        return (loss, scores, binary_labels)

    @staticmethod
    def make_compute_metrics(yes_token_id: int, no_token_id: int) -> Callable:
        """保留向下相容的工廠方法（不再使用，但避免外部呼叫報錯）。"""
        def _compute(eval_pred: EvalPrediction) -> dict:
            logits, labels = eval_pred
            scores = np.array(logits[:, 0] - logits[:, 1], dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
            return {
                "auc":     compute_auc(scores, labels),
                "hr@10":   compute_hr(scores, labels, k=10),
                "ndcg@10": compute_ndcg(scores, labels, k=10),
            }
        return _compute
