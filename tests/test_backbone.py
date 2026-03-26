# tests/test_backbone.py
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, call
from collm.model.backbone import build_backbone
from collm.training.config import BackboneConfig


@pytest.fixture
def backbone_config():
    return BackboneConfig(
        model_name_or_path="fake/model",
        lora_r=4,
        lora_alpha=8,
        lora_target_modules=["c_attn"],
        load_in_8bit=False,
    )


def test_build_backbone_registers_special_tokens(mocker, backbone_config):
    """build_backbone 必須注冊 [USER_TOKEN] 和 [ITEM_TOKEN]"""
    mock_model = MagicMock()
    mock_model.config.hidden_size = 16
    mock_tokenizer = MagicMock()
    mock_tokenizer.__len__ = MagicMock(return_value=32002)

    mocker.patch("collm.model.backbone.AutoModelForCausalLM.from_pretrained",
                 return_value=mock_model)
    mocker.patch("collm.model.backbone.AutoTokenizer.from_pretrained",
                 return_value=mock_tokenizer)
    mocker.patch("collm.model.backbone.get_peft_model", return_value=mock_model)

    build_backbone(backbone_config)

    mock_tokenizer.add_special_tokens.assert_called_once_with(
        {"additional_special_tokens": ["[USER_TOKEN]", "[ITEM_TOKEN]"]}
    )
    mock_model.resize_token_embeddings.assert_called_once_with(32002)


def test_build_backbone_applies_lora(mocker, backbone_config):
    """build_backbone 必須套用 LoRA"""
    mock_model = MagicMock()
    mock_model.config.hidden_size = 16
    mock_tokenizer = MagicMock()
    mock_tokenizer.__len__ = MagicMock(return_value=32000)
    mock_lora_model = MagicMock()

    mocker.patch("collm.model.backbone.AutoModelForCausalLM.from_pretrained",
                 return_value=mock_model)
    mocker.patch("collm.model.backbone.AutoTokenizer.from_pretrained",
                 return_value=mock_tokenizer)
    mock_get_peft = mocker.patch("collm.model.backbone.get_peft_model",
                                  return_value=mock_lora_model)

    model, tokenizer = build_backbone(backbone_config)

    assert mock_get_peft.called
    assert model is mock_lora_model


def test_build_backbone_returns_tuple(mocker, backbone_config):
    """返回值必須是 (model, tokenizer) tuple"""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.__len__ = MagicMock(return_value=32000)
    mocker.patch("collm.model.backbone.AutoModelForCausalLM.from_pretrained",
                 return_value=mock_model)
    mocker.patch("collm.model.backbone.AutoTokenizer.from_pretrained",
                 return_value=mock_tokenizer)
    mocker.patch("collm.model.backbone.get_peft_model", return_value=mock_model)

    result = build_backbone(backbone_config)
    assert isinstance(result, tuple)
    assert len(result) == 2
