from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
from collm.training.config import BackboneConfig
from collm.constants import USER_TOKEN, ITEM_TOKEN


def build_backbone(config: BackboneConfig) -> tuple:
    """載入 HuggingFace LLM、注冊占位符 token、套用 LoRA。

    占位符注冊確保 [USER_TOKEN] 和 [ITEM_TOKEN] 被 tokenize 為單一 token，
    使 DataCollator 能精確定位替換位置。

    Args:
        config: BackboneConfig，指定模型路徑和 LoRA 超參數

    Returns:
        (model, tokenizer) tuple。model 已套用 LoRA，tokenizer 已含占位符。
    """
    load_kwargs = dict(
        dtype=torch.float16,
        device_map="auto",
    )
    if config.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs.pop("dtype", None)  # 8bit 模式下不指定 dtype

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, **load_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, use_fast=False
    )

    # 注冊占位符 token（必須在 resize_token_embeddings 之前）
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [USER_TOKEN, ITEM_TOKEN]}
    )
    model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 套用 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
