"""
Rec Encoder 預訓練腳本（獨立於 CoLLM）。

用法：
    python scripts/train_rec_encoder.py \
        --data_path /data/ml-1m/train \
        --encoder_type sasrec \
        --embedding_dim 64 \
        --n_layers 2 \
        --n_heads 2 \
        --max_seq_len 50 \
        --epochs 50 \
        --lr 1e-3 \
        --save_path /checkpoints/sasrec_ml1m.pth
"""
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collm.training.config import RecEncoderConfig
from collm.encoders.mf import MFEncoder
from collm.encoders.sasrec import SASRecEncoder
from collm.encoders.din import DINEncoder


class PairDataset(Dataset):
    """用戶-物品交互對資料集（正樣本 + BPR 負採樣）。

    只使用 label=1 的正向互動作為正樣本，負樣本隨機採樣。
    """

    def __init__(self, df: pd.DataFrame, item_num: int, max_seq_len: int):
        # 只保留正向互動，避免把負向互動當成正樣本訓練
        self.records = df[df["label"] == 1].to_dict("records")
        self.item_num = item_num
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        neg_item = torch.randint(1, self.item_num, (1,)).item()
        hist = list(row.get("history_iid", []))[-self.max_seq_len:]
        padded = [0] * (self.max_seq_len - len(hist)) + hist
        return {
            "uid": int(row["uid"]),
            "pos_iid": int(row["iid"]),
            "neg_iid": neg_item,
            "seq_history": torch.tensor(padded, dtype=torch.long),
        }


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--encoder_type", default="sasrec",
                        choices=["mf", "sasrec", "din"])
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    df = pd.read_pickle(args.data_path + ".pkl")
    user_num = df["uid"].max() + 1
    item_num = df["iid"].max() + 1

    config = RecEncoderConfig(
        encoder_type=args.encoder_type,
        checkpoint_path="",
        embedding_dim=args.embedding_dim,
        user_num=int(user_num),
        item_num=int(item_num),
        max_seq_len=args.max_seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    if args.encoder_type == "mf":
        encoder = MFEncoder(config)
    elif args.encoder_type == "sasrec":
        encoder = SASRecEncoder(config)
    elif args.encoder_type == "din":
        encoder = DINEncoder(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    dataset = PairDataset(df, int(item_num), args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        encoder.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            uid = batch["uid"].to(device)
            pos = batch["pos_iid"].to(device)
            neg = batch["neg_iid"].to(device)
            seq = batch["seq_history"].to(device)

            # DIN 的 user representation 依賴 target item：
            # 對正樣本和負樣本分別計算各自的 user embedding
            if args.encoder_type == "din":
                user_emb_pos = encoder.get_user_embedding(uid, seq_history=seq, target_item_ids=pos)
                user_emb_neg = encoder.get_user_embedding(uid, seq_history=seq, target_item_ids=neg)
                pos_emb = encoder.get_item_embedding(pos)
                neg_emb = encoder.get_item_embedding(neg)
                pos_score = (user_emb_pos * pos_emb).sum(dim=-1)
                neg_score = (user_emb_neg * neg_emb).sum(dim=-1)
            else:
                user_emb = encoder.get_user_embedding(uid, seq_history=seq)
                pos_emb = encoder.get_item_embedding(pos)
                neg_emb = encoder.get_item_embedding(neg)
                pos_score = (user_emb * pos_emb).sum(dim=-1)
                neg_score = (user_emb * neg_emb).sum(dim=-1)

            loss = bpr_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")

    torch.save(encoder.state_dict(), args.save_path)
    print(f"Rec encoder saved to {args.save_path}")


if __name__ == "__main__":
    main()
