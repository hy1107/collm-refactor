"""
ML-1M 資料預處理腳本（Timestamp Split + Rolling History）。

切分策略（對應原始 CoLLM 論文）：
    1. 所有正向互動按全局時間戳排序
    2. 依 --train_ratio / --valid_ratio 將時間軸分成三段
    3. 訓練段內每一筆互動都生一個訓練樣本，
       樣本的 history = 該用戶在此互動之前的所有正向互動（遞增）
    4. Valid / Test 樣本只保留「目標物品是暖物品（出現在訓練正樣本集合）」的互動

每筆樣本欄位：
    uid, iid, title, history_iid, history_titles, label

預設 --train_ratio 0.06 / --valid_ratio 0.02 對應論文 Train~33K / Valid~10K / Test~7K。
執行時會先印出各段的原始互動數與估計樣本數，可依需求調整比例。

用法：
    python scripts/preprocess_ml1m.py \\
        --data_dir   /data/ml-1m \\
        --output_dir /data/ml-1m \\
        --train_ratio 0.06 \\
        --valid_ratio 0.02 \\
        --train_neg   1 \\
        --eval_neg    1 \\
        --min_inter   5
"""
import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd


def load_ratings(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "ratings.dat")
    return pd.read_csv(
        path, sep="::", header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
    )


def load_movies(data_dir: str) -> dict:
    path = os.path.join(data_dir, "movies.dat")
    df = pd.read_csv(
        path, sep="::", header=None,
        names=["MovieID", "Title", "Genres"],
        engine="python", encoding="latin-1",
    )
    return dict(zip(df["MovieID"], df["Title"]))


def make_record(uid, iid, title, history_iid, history_titles, label):
    return {
        "uid": uid, "iid": iid, "title": title,
        "history_iid": history_iid[:],
        "history_titles": history_titles[:],
        "label": label,
    }


def build_records(
    ratings: pd.DataFrame,
    id2title: dict,
    train_ratio: float,
    valid_ratio: float,
    train_neg: int,
    eval_neg: int,
    min_inter: int,
) -> tuple:
    # ── 1. 過濾正向互動，全局時間排序 ────────────────────────────────
    pos = ratings[ratings["Rating"] > 3].sort_values("Timestamp").reset_index(drop=True)

    # 重新編碼 ID 為 0-indexed
    user_ids = sorted(pos["UserID"].unique())
    item_ids = sorted(pos["MovieID"].unique())
    u2idx = {u: i for i, u in enumerate(user_ids)}
    i2idx = {it: i for i, it in enumerate(item_ids)}
    pos["uid"] = pos["UserID"].map(u2idx)
    pos["iid"] = pos["MovieID"].map(i2idx)

    # ── 2. 以時間戳決定三段切分點 ────────────────────────────────────
    ts_min = pos["Timestamp"].min()
    ts_max = pos["Timestamp"].max()
    ts_range = ts_max - ts_min

    train_end_ts = ts_min + int(ts_range * train_ratio)
    valid_end_ts = ts_min + int(ts_range * (train_ratio + valid_ratio))

    pos["split"] = "test"
    pos.loc[pos["Timestamp"] <  train_end_ts, "split"] = "train"
    pos.loc[(pos["Timestamp"] >= train_end_ts) &
            (pos["Timestamp"] <  valid_end_ts), "split"] = "valid"

    n_train_raw = (pos["split"] == "train").sum()
    n_valid_raw = (pos["split"] == "valid").sum()
    n_test_raw  = (pos["split"] == "test").sum()
    print(f"  時間戳切分：train={n_train_raw} / valid={n_valid_raw} / test={n_test_raw} 筆正向互動")

    # ── 3. 過濾互動次數不足的用戶 ────────────────────────────────────
    user_total = pos.groupby("uid").size()
    valid_users = set(user_total[user_total >= min_inter].index)
    pos = pos[pos["uid"].isin(valid_users)].reset_index(drop=True)

    # ── 4. 計算暖物品集合（訓練段出現過的目標物品）────────────────────
    train_items = set(pos[pos["split"] == "train"]["iid"])
    print(f"  暖物品數：{len(train_items)}")

    # ── 5. 逐時間順序處理，維護每位 user 的遞增歷史 ─────────────────
    all_items = list(range(len(item_ids)))
    user_interacted: dict[int, set] = defaultdict(set)
    user_hist_iid:   dict[int, list] = defaultdict(list)
    user_hist_title: dict[int, list] = defaultdict(list)

    train_records, valid_records, test_records = [], [], []

    for _, row in pos.iterrows():
        uid   = int(row["uid"])
        iid   = int(row["iid"])
        title = id2title.get(row["MovieID"], "Unknown")
        split = row["split"]

        hist_iid   = user_hist_iid[uid]
        hist_title = user_hist_title[uid]
        neg_pool   = list(set(all_items) - user_interacted[uid])

        if split == "train":
            # 訓練樣本：以當前歷史為 context，當前物品為 target
            train_records.append(make_record(uid, iid, title, hist_iid, hist_title, 1))
            for _ in range(train_neg):
                neg = random.choice(neg_pool) if neg_pool else iid
                train_records.append(make_record(
                    uid, neg, id2title.get(item_ids[neg], "Unknown"),
                    hist_iid, hist_title, 0
                ))

        elif split == "valid" and iid in train_items:
            # Valid：只保留目標是暖物品的樣本
            valid_records.append(make_record(uid, iid, title, hist_iid, hist_title, 1))
            for _ in range(eval_neg):
                neg = random.choice(neg_pool) if neg_pool else iid
                valid_records.append(make_record(
                    uid, neg, id2title.get(item_ids[neg], "Unknown"),
                    hist_iid, hist_title, 0
                ))

        elif split == "test" and iid in train_items:
            # Test：只保留目標是暖物品的樣本
            test_records.append(make_record(uid, iid, title, hist_iid, hist_title, 1))
            for _ in range(eval_neg):
                neg = random.choice(neg_pool) if neg_pool else iid
                test_records.append(make_record(
                    uid, neg, id2title.get(item_ids[neg], "Unknown"),
                    hist_iid, hist_title, 0
                ))

        # 更新歷史（無論哪個 split 都追蹤，使後續歷史完整）
        if iid not in user_interacted[uid]:
            user_hist_iid[uid].append(iid)
            user_hist_title[uid].append(title)
            user_interacted[uid].add(iid)

    return train_records, valid_records, test_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--train_ratio", type=float, default=0.06,
                        help="訓練段佔總時間範圍的比例（預設 0.06 ≈ 論文 ~33K）")
    parser.add_argument("--valid_ratio", type=float, default=0.02,
                        help="驗證段佔總時間範圍的比例（預設 0.02 ≈ 論文 ~10K）")
    parser.add_argument("--train_neg",   type=int, default=1,
                        help="訓練集負樣本數（預設 1）")
    parser.add_argument("--eval_neg",    type=int, default=1,
                        help="valid/test 負樣本數（預設 1）")
    parser.add_argument("--min_inter",   type=int, default=5,
                        help="用戶最少正向互動數（預設 5）")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("載入資料...")
    ratings  = load_ratings(args.data_dir)
    id2title = load_movies(args.data_dir)

    print("建構樣本...")
    train_rec, valid_rec, test_rec = build_records(
        ratings, id2title,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        train_neg=args.train_neg,
        eval_neg=args.eval_neg,
        min_inter=args.min_inter,
    )

    for name, records in [("train", train_rec), ("valid", valid_rec), ("test", test_rec)]:
        df = pd.DataFrame(records)
        out_path = os.path.join(args.output_dir, f"{name}.pkl")
        df.to_pickle(out_path)
        pos = (df["label"] == 1).sum()
        neg = (df["label"] == 0).sum()
        print(f"  {name}.pkl: {len(df):>7} 筆（pos={pos}, neg={neg}）→ {out_path}")

    print("完成。")


if __name__ == "__main__":
    main()
