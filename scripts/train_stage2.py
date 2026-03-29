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
import numpy as np
import pandas as pd
import torch
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


def build_lightgcn_adj(df: pd.DataFrame, user_num: int, item_num: int) -> torch.Tensor:
    """從互動資料建構 LightGCN 正規化鄰接矩陣（D^{-1/2} A D^{-1/2}）。

    只使用正向互動（label=1）建構圖。
    User 節點 index：0 ~ user_num-1
    Item 節點 index：user_num ~ user_num+item_num-1
    """
    pos = df[df["label"] == 1]
    users = pos["uid"].values
    items = pos["iid"].values + user_num   # item index 偏移到 user 之後

    n = user_num + item_num
    # 雙向邊：user→item 和 item→user
    row = np.concatenate([users, items])
    col = np.concatenate([items, users])
    data = np.ones(len(row), dtype=np.float32)

    # 計算 degree，做 D^{-1/2} 正規化
    deg = np.zeros(n, dtype=np.float32)
    np.add.at(deg, row, 1.0)
    deg_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)

    norm_data = deg_inv_sqrt[row] * data * deg_inv_sqrt[col]

    indices = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    values = torch.tensor(norm_data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n))


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
    if cfg.rec.encoder_type == "lightgcn":
        adj_matrix = build_lightgcn_adj(train_df, cfg.rec.user_num, cfg.rec.item_num)
        rec_encoder = LightGCNEncoder.from_pretrained(cfg.rec.checkpoint_path, cfg.rec, adj_matrix=adj_matrix)
    else:
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

    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode(" No",  add_special_tokens=False)[0]

    trainer = CoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=CoLLMTrainer.make_compute_metrics(yes_id, no_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Stage 2 完成，checkpoint 已存至 {args.output_dir}")


if __name__ == "__main__":
    main()
