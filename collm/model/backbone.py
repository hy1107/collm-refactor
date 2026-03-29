import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import torch
from collm.training.config import BackboneConfig
from collm.constants import USER_TOKEN, ITEM_TOKEN


def build_backbone(config: BackboneConfig) -> tuple:
    """載入 HuggingFace LLM、注冊占位符 token、套用 LoRA。

    若 model_name_or_path 含 adapter_config.json（Stage 1 checkpoint），
    則用 PeftModel.from_pretrained 載入已訓練的 LoRA，不重新初始化。

    Args:
        config: BackboneConfig，指定模型路徑和 LoRA 超參數

    Returns:
        (model, tokenizer) tuple。model 已套用 LoRA，tokenizer 已含占位符。
    """
    is_peft_checkpoint = os.path.exists(
        os.path.join(config.model_name_or_path, "adapter_config.json")
    )

    load_kwargs = dict(
        dtype=torch.float16,
        device_map="auto",
    )
    if config.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs.pop("dtype", None)

    if is_peft_checkpoint:
        # Stage 2：從 adapter_config.json 取得 base model 路徑，再套上 LoRA
        import json
        adapter_cfg = json.load(
            open(os.path.join(config.model_name_or_path, "adapter_config.json"))
        )
        base_model_path = adapter_cfg["base_model_name_or_path"]

        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

        tokenizer.add_special_tokens(
            {"additional_special_tokens": [USER_TOKEN, ITEM_TOKEN]}
        )
        base_model.resize_token_embeddings(len(tokenizer))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = PeftModel.from_pretrained(base_model, config.model_name_or_path)
        model.print_trainable_parameters()
    else:
        # Stage 1：載入 base model，初始化新 LoRA
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, **load_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, use_fast=False
        )

        tokenizer.add_special_tokens(
            {"additional_special_tokens": [USER_TOKEN, ITEM_TOKEN]}
        )
        model.resize_token_embeddings(len(tokenizer))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
