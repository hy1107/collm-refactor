import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from safetensors.torch import load_file
import torch
from collm.training.config import BackboneConfig
from collm.constants import USER_TOKEN, ITEM_TOKEN


def build_backbone(config: BackboneConfig) -> tuple:
    """載入 HuggingFace LLM、注冊占位符 token、套用 LoRA。

    三種情況：
    1. adapter_config.json 存在 → PEFT checkpoint，用 PeftModel 載入
    2. config.json + model.safetensors 存在（CoLLMModel state dict）
       → 從 config.json 取得 base model，套 LoRA，再從 state dict 還原 backbone 權重
    3. 都不存在 → base model，初始化新 LoRA（Stage 1 訓練）
    """
    path = config.model_name_or_path
    is_peft_checkpoint = (
        os.path.exists(os.path.join(path, "adapter_config.json")) and
        (os.path.exists(os.path.join(path, "adapter_model.safetensors")) or
         os.path.exists(os.path.join(path, "adapter_model.bin")))
    )
    is_collm_checkpoint = (
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

    def _add_tokens(model, tokenizer):
        tokenizer.add_special_tokens({"additional_special_tokens": [USER_TOKEN, ITEM_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if is_peft_checkpoint:
        adapter_cfg = json.load(open(os.path.join(path, "adapter_config.json")))
        base_path = adapter_cfg["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
        _add_tokens(base_model, tokenizer)
        model = PeftModel.from_pretrained(base_model, path)
        print(f"[backbone] PEFT checkpoint: {path}")

    elif is_collm_checkpoint:
        # Stage 1 存了整個 CoLLMModel state dict（鍵名帶 backbone. 前綴）
        # 從 config.json 取 base model 路徑，套上同樣的 LoRA，再還原權重
        cfg_json = json.load(open(os.path.join(path, "config.json")))
        base_path = cfg_json.get("_name_or_path", "lmsys/vicuna-7b-v1.3")

        base_model = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
        _add_tokens(base_model, tokenizer)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)

        # 從 CoLLMModel state dict 提取 backbone 權重（去掉 backbone. 前綴）
        full_state = load_file(os.path.join(path, "model.safetensors"))
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in full_state.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        print(f"[backbone] CoLLMModel checkpoint loaded, missing={len(missing)}, unexpected={len(unexpected)}")

    else:
        # Base model + 初始化新 LoRA（Stage 1 訓練）
        base_model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        _add_tokens(base_model, tokenizer)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        print(f"[backbone] base model + new LoRA: {path}")

    return model, tokenizer
