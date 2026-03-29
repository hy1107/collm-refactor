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

    三種工作模式：
    - Stage 1（rec_encoder=None）：純語言模型，直接 forward backbone
    - Stage 2 / injection_layer=0：替換 input embedding 層的佔位符後再 forward（原始 CoLLM）
    - Stage 2 / injection_layer=N（N≥1）：透過 forward hook 在第 N 個 transformer block
      的輸出後替換對應位置的 hidden states

    Args:
        backbone:         HuggingFace LLM（已套用 LoRA）
        tokenizer:        對應的 tokenizer（已注冊占位符 token）
        rec_encoder:      推薦 Encoder（Stage 1 時為 None）
        cie_module:       CIE 投影層（Stage 1 時為 None）
        config:           CoLLMConfig
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
        self.injection_layer = config.injection_layer if config is not None else 0

    # ------------------------------------------------------------------
    # 輔助：取得 backbone 的 transformer block 列表
    # 相容 PEFT 包裝（PeftModel → LoraModel → LlamaForCausalLM → LlamaModel）
    # ------------------------------------------------------------------
    def _get_transformer_layers(self) -> nn.ModuleList:
        m = self.backbone
        # 逐層 unwrap：PEFT wrapper → inner model → decoder stack
        for attr in ("base_model", "model", "model"):
            if hasattr(m, attr):
                m = getattr(m, attr)
        for attr in ("layers", "h", "blocks"):
            if hasattr(m, attr):
                return getattr(m, attr)
        raise AttributeError(
            "無法從 backbone 取得 transformer layers，"
            "請確認模型架構（支援 LLaMA/Mistral/GPT-2/GPT-NeoX）"
        )

    # ------------------------------------------------------------------
    # 核心 forward
    # ------------------------------------------------------------------
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

        need_inject = (
            self.rec_encoder is not None
            and self.cie_module is not None
            and self.use_collaborative_signal
            and user_ids is not None
            and target_item_ids is not None
        )

        if not need_inject:
            return self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        # 計算協同 token（兩種注入模式共用）
        user_emb = self.rec_encoder.get_user_embedding(
            user_ids, seq_history=seq_history, target_item_ids=target_item_ids,
        )
        item_emb = self.rec_encoder.get_item_embedding(
            target_item_ids, seq_history=seq_history,
        )
        user_token = self.cie_module(user_emb)   # (batch, d_llm)
        item_token = self.cie_module(item_emb)   # (batch, d_llm)

        if self.injection_layer == 0:
            # ── 模式 A：替換 input embedding ──────────────────────────
            self._replace_positions(inputs_embeds, user_token, item_token,
                                    user_placeholder_pos, item_placeholder_pos)
            return self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            # ── 模式 B：在第 injection_layer 個 block 輸出後注入 ──────
            layers = self._get_transformer_layers()
            target_idx = self.injection_layer - 1  # 0-indexed
            if target_idx >= len(layers):
                raise ValueError(
                    f"injection_layer={self.injection_layer} 超過模型總層數 {len(layers)}"
                )

            def _hook(module, inputs, output):
                # transformer block 輸出通常是 tuple，第 0 個是 hidden states
                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                self._replace_positions(hidden, user_token, item_token,
                                        user_placeholder_pos, item_placeholder_pos)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

            handle = layers[target_idx].register_forward_hook(_hook)
            try:
                result = self.backbone(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            finally:
                handle.remove()  # 確保 hook 一定被移除，避免影響下次 forward
            return result

    # ------------------------------------------------------------------
    # 共用：就地替換指定位置的向量
    # ------------------------------------------------------------------
    @staticmethod
    def _replace_positions(
        tensor: Tensor,
        user_token: Tensor,
        item_token: Tensor,
        user_placeholder_pos: Optional[Tensor],
        item_placeholder_pos: Optional[Tensor],
    ) -> None:
        if user_placeholder_pos is not None:
            for i in range(tensor.size(0)):
                if user_placeholder_pos[i] != -1:
                    tensor[i, user_placeholder_pos[i]] = user_token[i]
        if item_placeholder_pos is not None:
            for i in range(tensor.size(0)):
                if item_placeholder_pos[i] != -1:
                    tensor[i, item_placeholder_pos[i]] = item_token[i]

    def freeze_backbone(self) -> None:
        """凍結 backbone 所有參數（含 LoRA）"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_rec_encoder(self) -> None:
        """凍結 rec encoder 所有參數"""
        if self.rec_encoder is not None:
            self.rec_encoder.freeze()
