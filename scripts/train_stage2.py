"""
Stage 2：CIE 訓練（帶協同信號）。
載入 Stage 1 checkpoint，凍結 backbone + LoRA + rec encoder，只訓練 CIE 投影層。

用法：
    python scripts/train_stage2.py \
        --config configs/stage2_movielens.yaml \
        --stage1_checkpoint /checkpoints/stage1 \
        --output_dir /checkpoints/stage2 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 4 \
        --learning_rate 1e-4
"""
import argparse
import yaml
import dacite
import pandas as pd
from transformers import TrainingArguments

from collm.training.config import CoLLMConfig, RecEncoderConfig
from collm.model.backbone import build_backbone
from collm.model.cie import CIEModule
from collm.model.collm import CoLLMModel
from collm.data.dataset import RecDataset
from collm.data.collator import RecDataCollator
from collm.training.trainer import CoLLMTrainer
from collm.encoders.mf import MFEncoder
from collm.encoders.lightgcn import LightGCNEncoder
from collm.encoders.sasrec import SASRecEncoder
from collm.encoders.din import DINEncoder


_ENCODER_MAP = {
    "mf": MFEncoder,
    "lightgcn": LightGCNEncoder,
    "sasrec": SASRecEncoder,
    "din": DINEncoder,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1_checkpoint", required=True,
                        help="Stage 1 存下的 checkpoint 目錄")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=50)
    args = parser.parse_args()

    raw = yaml.safe_load(open(args.config))
    cfg = dacite.from_dict(CoLLMConfig, raw, config=dacite.Config(cast=[list]))

    train_df = pd.read_pickle(cfg.data.data_path + "/train.pkl")
    cfg.rec.user_num = int(train_df["uid"].max()) + 1
    cfg.rec.item_num = int(train_df["iid"].max()) + 1
    cfg.rec.use_collaborative_signal = True

    cfg.backbone.model_name_or_path = args.stage1_checkpoint
    backbone, tokenizer = build_backbone(cfg.backbone)

    EncoderClass = _ENCODER_MAP[cfg.rec.encoder_type]
    rec_encoder = EncoderClass.from_pretrained(cfg.rec.checkpoint_path, cfg.rec)

    base_config = getattr(backbone, "base_model", backbone).config
    llm_dim = base_config.hidden_size
    cie = CIEModule(rec_dim=cfg.rec.embedding_dim, llm_dim=llm_dim)

    model = CoLLMModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rec_encoder=rec_encoder,
        cie_module=cie,
        config=cfg,
    )
    model.freeze_backbone()
    model.freeze_rec_encoder()

    train_ds = RecDataset(cfg.data.data_path + "/train", cfg.data.dataset_type)
    eval_ds = RecDataset(cfg.data.data_path + "/valid", cfg.data.dataset_type)

    is_sequential = cfg.rec.encoder_type in ("sasrec", "din")
    collator = RecDataCollator(
        tokenizer=tokenizer,
        use_collaborative_signal=True,
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
        fp16=True,
        remove_unused_columns=False,
    )

    trainer = CoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Stage 2 完成，checkpoint 已存至 {args.output_dir}")


if __name__ == "__main__":
    main()
