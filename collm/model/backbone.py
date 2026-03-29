import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import torch
from collm.training.config import BackboneConfig
from collm.constants import USER_TOKEN, ITEM_TOKEN


def build_backbone(config: BackboneConfig) -> tuple:
    """載入 HuggingFace LLM、注冊占位符 token、套用 LoRA。

    三種情況：
    1. adapter_config.json 存在 → PEFT checkpoint，用 PeftModel 載入
    2. config.json 存在（無 adapter_config.json）→ 全量 Stage 1 checkpoint，直接載入，不套新 LoRA
    3. 都不存在 → base model，初始化新 LoRA（Stage 1 訓練）

    Args:
        config: BackboneConfig，指定模型路徑和 LoRA 超參數

    Returns:
        (model, tokenizer) tuple。
    """
    path = config.model_name_or_path
    is_peft_checkpoint = os.path.exists(os.path.join(path, "adapter_config.json"))
    is_full_checkpoint = (
        not is_peft_checkpoint
        and os.path.exists(os.path.join(path, "config.json"))
        and os.path.exists(os.path.join(path, "model.safetensors"))
    )

    load_kwargs = dict(
        dtype=torch.float16,
        device_map="auto",
    )
    if config.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs.pop("dtype", None)

    if is_peft_checkpoint:
        # PEFT adapter checkpoint（理想的 Stage 1 存法）
        import json
        adapter_cfg = json.load(open(os.path.join(path, "adapter_config.json")))
        base_path = adapter_cfg["base_model_name_or_path"]

        base_model = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
        tokenizer.add_special_tokens({"additional_special_tokens": [USER_TOKEN, ITEM_TOKEN]})
        base_model.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = PeftModel.from_pretrained(base_model, path)
        print(f"[backbone] 載入 PEFT checkpoint：{path}")

    elif is_full_checkpoint:
        # 全量 Stage 1 checkpoint（LoRA 已合併）：直接載入，不套新 LoRA
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        tokenizer.add_special_tokens({"additional_special_tokens": [USER_TOKEN, ITEM_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"[backbone] 載入全量 Stage 1 checkpoint：{path}")

    else:
        # Base model：初始化新 LoRA（Stage 1 訓練用）
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        tokenizer.add_special_tokens({"additional_special_tokens": [USER_TOKEN, ITEM_TOKEN]})
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
        print(f"[backbone] 載入 base model + 初始化 LoRA：{path}")

    return model, tokenizer
