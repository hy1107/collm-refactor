from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from collm.encoders.base import BaseRecEncoder
from collm.model.cie import CIEModule
from collm.training.config import CoLLMConfig


class CoLLMModel(nn.Module):
    """CoLLM 主模型：將協同過濾信號注入 LLM 的 token embedding 空間。

    兩種工作模式：
    - Stage 1（rec_encoder=None）：純語言模型，直接 forward backbone
    - Stage 2（rec_encoder 非 None + use_collaborative_signal=True）：
      替換 [USER_TOKEN] 和 [ITEM_TOKEN] 位置的 embedding 後再 forward

    Args:
        backbone:       HuggingFace LLM（已套用 LoRA）
        tokenizer:      對應的 tokenizer（已注冊占位符 token）
        rec_encoder:    推薦 Encoder（Stage 1 時為 None）
        cie_module:     CIE 投影層（Stage 1 時為 None）
        config:         CoLLMConfig
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        rec_encoder: Optional[BaseRecEncoder] = None,
        cie_module: Optional[CIEModule] = None,
        config: Optional[CoLLMConfig] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.rec_encoder = rec_encoder
        self.cie_module = cie_module
        self.use_collaborative_signal = (
            config.rec.use_collaborative_signal if config is not None else False
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        user_ids: Optional[Tensor] = None,
        target_item_ids: Optional[Tensor] = None,
        user_placeholder_pos: Optional[Tensor] = None,
        item_placeholder_pos: Optional[Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        所有參數均為 LongTensor 或 FloatTensor，相容 HF Trainer 的 device 轉移。
        seq_history 透過 **kwargs 傳入（僅序列模型批次含此鍵）。
        """
        seq_history: Optional[Tensor] = kwargs.get("seq_history")

        embed_fn = self.backbone.get_input_embeddings()
        inputs_embeds = embed_fn(input_ids).clone()  # (batch, seq, d_llm)

        if (
            self.rec_encoder is not None
            and self.cie_module is not None
            and self.use_collaborative_signal
            and user_ids is not None
            and target_item_ids is not None
        ):
            user_emb = self.rec_encoder.get_user_embedding(
                user_ids,
                seq_history=seq_history,
                target_item_ids=target_item_ids,
            )
            item_emb = self.rec_encoder.get_item_embedding(
                target_item_ids,
                seq_history=seq_history,
            )

            user_token = self.cie_module(user_emb)
            item_token = self.cie_module(item_emb)

            if user_placeholder_pos is not None:
                for i in range(inputs_embeds.size(0)):
                    if user_placeholder_pos[i] != -1:
                        inputs_embeds[i, user_placeholder_pos[i]] = user_token[i]

            if item_placeholder_pos is not None:
                for i in range(inputs_embeds.size(0)):
                    if item_placeholder_pos[i] != -1:
                        inputs_embeds[i, item_placeholder_pos[i]] = item_token[i]

        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def freeze_backbone(self) -> None:
        """凍結 backbone 所有參數（含 LoRA）"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_rec_encoder(self) -> None:
        """凍結 rec encoder 所有參數"""
        if self.rec_encoder is not None:
            self.rec_encoder.freeze()
