# tests/test_collm_model.py
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from collm.model.collm import CoLLMModel
from collm.model.cie import CIEModule
from collm.encoders.mf import MFEncoder
from collm.training.config import RecEncoderConfig, CoLLMConfig, BackboneConfig, DataConfig


# ── TinyLM：模擬 HuggingFace LLM 介面 ──────────────────────────────────
class TinyLM(nn.Module):
    def __init__(self, vocab_size=100, d_model=16):
        super().__init__()
        self._embeddings = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def get_input_embeddings(self):
        return self._embeddings

    def forward(self, input_ids=None, inputs_embeds=None,
                attention_mask=None, labels=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self._embeddings(input_ids)
        logits = self.lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def config():
    return CoLLMConfig(
        rec=RecEncoderConfig(
            encoder_type="mf", checkpoint_path="",
            embedding_dim=8, use_collaborative_signal=True,
            user_num=10, item_num=20,
        ),
        backbone=BackboneConfig(model_name_or_path=""),
        data=DataConfig(data_path="", dataset_type="movielens"),
    )


@pytest.fixture
def config_no_signal(config):
    config.rec.use_collaborative_signal = False
    return config


@pytest.fixture
def tiny_lm():
    return TinyLM(vocab_size=100, d_model=16)


@pytest.fixture
def tiny_tokenizer_mock():
    class FakeTok:
        pass
    return FakeTok()


# ── Tests ───────────────────────────────────────────────────────────────
def test_stage1_forward_no_encoder(tiny_lm, config_no_signal, tiny_tokenizer_mock):
    """Stage 1：無 rec_encoder/cie，純語言模型 forward"""
    model = CoLLMModel(
        backbone=tiny_lm,
        tokenizer=tiny_tokenizer_mock,
        config=config_no_signal,
    )
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :5] = -100

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0


def test_stage2_forward_with_encoder(tiny_lm, config, tiny_tokenizer_mock):
    """Stage 2：有 rec_encoder + cie，embedding 替換後 forward"""
    rec_encoder = MFEncoder(config.rec)
    cie = CIEModule(rec_dim=8, llm_dim=16)

    model = CoLLMModel(
        backbone=tiny_lm,
        tokenizer=tiny_tokenizer_mock,
        rec_encoder=rec_encoder,
        cie_module=cie,
        config=config,
    )
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :8] = -100
    user_ids = torch.tensor([0, 1])
    target_item_ids = torch.tensor([5, 6])
    user_ph_pos = torch.tensor([2, 2])
    item_ph_pos = torch.tensor([5, 5])

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        user_ids=user_ids,
        target_item_ids=target_item_ids,
        user_placeholder_pos=user_ph_pos,
        item_placeholder_pos=item_ph_pos,
    )
    assert out.loss is not None


def test_stage2_embedding_substitution(tiny_lm, config, tiny_tokenizer_mock):
    """確認 embedding 確實在正確位置被替換"""
    from unittest.mock import patch

    rec_encoder = MFEncoder(config.rec)
    cie = CIEModule(rec_dim=8, llm_dim=16)
    model = CoLLMModel(
        backbone=tiny_lm,
        tokenizer=tiny_tokenizer_mock,
        rec_encoder=rec_encoder,
        cie_module=cie,
        config=config,
    )

    input_ids = torch.zeros(1, 8, dtype=torch.long)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    user_ids = torch.tensor([0])
    target_item_ids = torch.tensor([5])
    user_ph_pos = torch.tensor([2])
    item_ph_pos = torch.tensor([5])

    orig_emb = tiny_lm.get_input_embeddings()(input_ids).detach().clone()

    captured = {}
    original_forward = tiny_lm.__class__.forward

    def patched_forward(self, input_ids=None, inputs_embeds=None,
                        attention_mask=None, labels=None, **kw):
        captured["inputs_embeds"] = inputs_embeds.detach().clone() if inputs_embeds is not None else None
        return original_forward(self, input_ids=input_ids,
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                labels=labels, **kw)

    with patch.object(tiny_lm.__class__, "forward", patched_forward):
        model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            user_ids=user_ids, target_item_ids=target_item_ids,
            user_placeholder_pos=user_ph_pos, item_placeholder_pos=item_ph_pos,
        )

    assert captured["inputs_embeds"] is not None
    assert not torch.allclose(captured["inputs_embeds"][0, 2], orig_emb[0, 2])


def test_neg1_pos_skips_substitution(tiny_lm, config, tiny_tokenizer_mock):
    """-1 位置不應替換 embedding，所有位置的 embedding 應與原始一致"""
    from unittest.mock import patch

    rec_encoder = MFEncoder(config.rec)
    cie = CIEModule(rec_dim=8, llm_dim=16)
    model = CoLLMModel(
        backbone=tiny_lm, tokenizer=tiny_tokenizer_mock,
        rec_encoder=rec_encoder, cie_module=cie, config=config,
    )
    input_ids = torch.zeros(1, 8, dtype=torch.long)
    orig_emb = tiny_lm.get_input_embeddings()(input_ids).detach().clone()

    captured = {}
    original_forward = tiny_lm.__class__.forward

    def patched_forward(self, input_ids=None, inputs_embeds=None,
                        attention_mask=None, labels=None, **kw):
        captured["inputs_embeds"] = inputs_embeds.detach().clone() if inputs_embeds is not None else None
        return original_forward(self, input_ids=input_ids,
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                labels=labels, **kw)

    with patch.object(tiny_lm.__class__, "forward", patched_forward):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            labels=torch.full((1, 8), -100, dtype=torch.long),
            user_ids=torch.tensor([0]),
            target_item_ids=torch.tensor([5]),
            user_placeholder_pos=torch.tensor([-1]),
            item_placeholder_pos=torch.tensor([-1]),
        )

    assert captured["inputs_embeds"] is not None
    assert torch.allclose(captured["inputs_embeds"], orig_emb)
