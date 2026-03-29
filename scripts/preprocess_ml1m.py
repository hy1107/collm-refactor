"""
ML-1M 資料預處理腳本（Timestamp Split + Rolling History）。

對應原始 CoLLM 論文做法：
    - 使用「所有」評分（不只正向）
    - rating > 3 → label=1（正），其餘 → label=0（負）
    - History 只由正向互動（rating > 3）遞增累積
    - 全局時間戳切分成訓練 / 驗證 / 測試三段
    - 每筆落在訓練段的評分都生一個訓練樣本
    - Valid / Test 只保留目標物品是暖物品（出現在訓練正樣本）的互動

每筆樣本欄位：
    uid, iid, title, history_iid, history_titles, label

用法：
    python scripts/preprocess_ml1m.py \\
        --data_dir   /data/ml-1m \\
        --output_dir /data/ml-1m \\
        --train_ratio 0.033 \\
        --valid_ratio 0.010
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
    min_inter: int,
) -> tuple:
    # ── 1. 所有評分按時間排序（正負都保留）────────────────────────────
    all_ratings = ratings.sort_values("Timestamp").reset_index(drop=True)

    # 重新編碼 ID 為 0-indexed（以所有出現過的 user/item 為基礎）
    user_ids = sorted(all_ratings["UserID"].unique())
    item_ids = sorted(all_ratings["MovieID"].unique())
    u2idx = {u: i for i, u in enumerate(user_ids)}
    i2idx = {it: i for i, it in enumerate(item_ids)}
    all_ratings["uid"] = all_ratings["UserID"].map(u2idx)
    all_ratings["iid"] = all_ratings["MovieID"].map(i2idx)
    all_ratings["label"] = (all_ratings["Rating"] > 3).astype(int)

    # ── 2. 以時間戳決定三段切分點 ────────────────────────────────────
    ts_min = all_ratings["Timestamp"].min()
    ts_max = all_ratings["Timestamp"].max()
    ts_range = ts_max - ts_min

    train_end_ts = ts_min + int(ts_range * train_ratio)
    valid_end_ts = ts_min + int(ts_range * (train_ratio + valid_ratio))

    all_ratings["split"] = "test"
    all_ratings.loc[all_ratings["Timestamp"] < train_end_ts, "split"] = "train"
    all_ratings.loc[
        (all_ratings["Timestamp"] >= train_end_ts) &
        (all_ratings["Timestamp"] <  valid_end_ts), "split"
    ] = "valid"

    for s in ["train", "valid", "test"]:
        n = (all_ratings["split"] == s).sum()
        pos = ((all_ratings["split"] == s) & (all_ratings["label"] == 1)).sum()
        print(f"  {s}: {n} 筆（pos={pos}, neg={n-pos}）")

    # ── 3. 過濾互動次數不足的用戶（以正向互動計）────────────────────
    pos_counts = all_ratings[all_ratings["label"] == 1].groupby("uid").size()
    valid_users = set(pos_counts[pos_counts >= min_inter].index)
    all_ratings = all_ratings[all_ratings["uid"].isin(valid_users)].reset_index(drop=True)

    # ── 4. 計算暖物品集合（訓練段的正樣本目標）──────────────────────
    train_pos = all_ratings[(all_ratings["split"] == "train") & (all_ratings["label"] == 1)]
    warm_items = set(train_pos["iid"])
    print(f"  暖物品數：{len(warm_items)}")

    # ── 5. 逐時間順序建樣本，維護各 user 的正向歷史 ─────────────────
    user_hist_iid:   dict[int, list] = defaultdict(list)
    user_hist_title: dict[int, list] = defaultdict(list)
    user_pos_seen:   dict[int, set]  = defaultdict(set)

    train_records, valid_records, test_records = [], [], []

    for _, row in all_ratings.iterrows():
        uid   = int(row["uid"])
        iid   = int(row["iid"])
        label = int(row["label"])
        title = id2title.get(row["MovieID"], "Unknown")
        split = row["split"]

        hist_iid   = user_hist_iid[uid]
        hist_title = user_hist_title[uid]

        record = make_record(uid, iid, title, hist_iid, hist_title, label)

        if split == "train":
            train_records.append(record)

        elif split == "valid" and iid in warm_items:
            valid_records.append(record)

        elif split == "test" and iid in warm_items:
            test_records.append(record)

        # 只有正向互動才加入歷史
        if label == 1 and iid not in user_pos_seen[uid]:
            user_hist_iid[uid].append(iid)
            user_hist_title[uid].append(title)
            user_pos_seen[uid].add(iid)

    return train_records, valid_records, test_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--train_ratio", type=float, default=0.033,
                        help="訓練段佔總時間範圍的比例（預設 0.033 ≈ 論文 Train~33K）")
    parser.add_argument("--valid_ratio", type=float, default=0.010,
                        help="驗證段佔總時間範圍的比例（預設 0.010 ≈ 論文 Valid~10K）")
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

    print("切分統計（過濾前）：")
    train_rec, valid_rec, test_rec = build_records(
        ratings, id2title,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        min_inter=args.min_inter,
    )

    print("儲存...")
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
