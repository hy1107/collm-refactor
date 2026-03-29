"""
Stage 1：LoRA 預訓練（純文字，無協同信號）。

用法：
    python scripts/train_stage1.py \
        --config configs/stage1_movielens.yaml \
        --output_dir /checkpoints/stage1 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --learning_rate 2e-4
"""
import argparse
import yaml
import dacite
import pandas as pd
from transformers import TrainingArguments
from torch.utils.data import DataLoader

from collm.training.config import CoLLMConfig
from collm.model.backbone import build_backbone
from collm.model.collm import CoLLMModel
from collm.data.dataset import RecDataset
from collm.data.collator import RecDataCollator
from collm.training.trainer import CoLLMTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="CoLLMConfig YAML 路徑")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    raw = yaml.safe_load(open(args.config))
    cfg = dacite.from_dict(CoLLMConfig, raw, config=dacite.Config(cast=[list]))

    train_df = pd.read_pickle(cfg.data.data_path + "/train.pkl")
    cfg.rec.user_num = int(train_df["uid"].max()) + 1
    cfg.rec.item_num = int(train_df["iid"].max()) + 1

    cfg.rec.use_collaborative_signal = False

    backbone, tokenizer = build_backbone(cfg.backbone)
    model = CoLLMModel(backbone, tokenizer, config=cfg)

    train_ds = RecDataset(cfg.data.data_path + "/train", cfg.data.dataset_type)
    eval_ds = RecDataset(cfg.data.data_path + "/valid", cfg.data.dataset_type)

    is_sequential = cfg.rec.encoder_type in ("sasrec", "din")
    collator = RecDataCollator(
        tokenizer=tokenizer,
        use_collaborative_signal=False,
        max_seq_len=cfg.data.max_seq_len,
        max_history_len=cfg.data.max_history_len,
        is_sequential=is_sequential,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,
        remove_unused_columns=False,
    )

    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode(" No",  add_special_tokens=False)[0]

    trainer = CoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        yes_token_id=yes_id,
        no_token_id=no_id,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # 明確存 PEFT adapter（產生 adapter_config.json + adapter_model.safetensors）
    model.backbone.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Stage 1 完成，checkpoint 已存至 {args.output_dir}")


if __name__ == "__main__":
    main()
