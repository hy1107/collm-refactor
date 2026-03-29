"""
ML-1M 資料預處理腳本。

產生 train.pkl / valid.pkl / test.pkl，格式與 RecDataset 相容。

每筆樣本欄位：
    uid           : int，用戶 ID（0-indexed）
    iid           : int，物品 ID（0-indexed）
    title         : str，電影標題
    history_iid   : list[int]，歷史互動物品 ID（時間升序）
    history_titles: list[str]，歷史互動物品標題
    label         : int，1=正樣本，0=負樣本

切分策略（leave-one-out）：
    每位用戶按時間排序後，最後一筆 → test，倒數第二筆 → valid，其餘 → train
    負樣本：對每筆正樣本隨機採樣一筆用戶未互動過的物品

用法：
    python scripts/preprocess_ml1m.py \
        --data_dir /data/ml-1m \
        --output_dir /data/ml-1m \
        --neg_per_pos 1 \
        --min_interactions 5
"""
import argparse
import os
import random
import pandas as pd
import numpy as np


def load_ratings(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path, sep="::", header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
    )
    return df


def load_movies(data_dir: str) -> dict:
    path = os.path.join(data_dir, "movies.dat")
    df = pd.read_csv(
        path, sep="::", header=None,
        names=["MovieID", "Title", "Genres"],
        engine="python", encoding="latin-1",
    )
    return dict(zip(df["MovieID"], df["Title"]))


def build_records(ratings: pd.DataFrame, id2title: dict, neg_per_pos: int) -> tuple:
    """回傳 (train_records, valid_records, test_records)"""
    # 只保留正向互動（rating ≥ 4），按時間排序
    pos = ratings[ratings["Rating"] >= 4].sort_values("Timestamp")

    # 重新編碼為 0-indexed
    user_ids = sorted(pos["UserID"].unique())
    item_ids = sorted(pos["MovieID"].unique())
    u2idx = {u: i for i, u in enumerate(user_ids)}
    i2idx = {it: i for i, it in enumerate(item_ids)}

    # 每個 user 的正向互動序列
    user_history: dict[int, list] = {}
    for _, row in pos.iterrows():
        uid = u2idx[row["UserID"]]
        iid = i2idx[row["MovieID"]]
        user_history.setdefault(uid, [])
        if iid not in user_history[uid]:  # 去重（保留第一次）
            user_history[uid].append(iid)

    all_items = set(range(len(item_ids)))
    train_records, valid_records, test_records = [], [], []

    for uid, hist in user_history.items():
        if len(hist) < 3:
            continue  # 至少需要 3 筆才能切 train/valid/test

        # 各筆 positive 的 title 列表
        titles = [id2title.get(item_ids[iid], "Unknown") for iid in hist]

        # leave-one-out 切分
        # test = 最後一筆，valid = 倒數第二筆，train = 其餘
        splits = [
            (test_records,  hist[:-1],  hist[-1],  titles[:-1],  titles[-1]),
            (valid_records, hist[:-2],  hist[-2],  titles[:-2],  titles[-2]),
        ]
        for i, iid in enumerate(hist[:-2]):
            splits.append((train_records, hist[:i], iid, titles[:i], titles[i]))

        neg_pool = list(all_items - set(hist))

        for target_list, history, target_iid, history_titles, target_title in splits:
            # 正樣本
            target_list.append({
                "uid": uid, "iid": target_iid,
                "title": target_title,
                "history_iid": history[:],
                "history_titles": history_titles[:],
                "label": 1,
            })
            # 負樣本
            for _ in range(neg_per_pos):
                neg_iid = random.choice(neg_pool)
                target_list.append({
                    "uid": uid, "iid": neg_iid,
                    "title": id2title.get(item_ids[neg_iid], "Unknown"),
                    "history_iid": history[:],
                    "history_titles": history_titles[:],
                    "label": 0,
                })

    return train_records, valid_records, test_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True, help="ml-1m 原始資料目錄")
    parser.add_argument("--output_dir", required=True, help="pkl 輸出目錄（可與 data_dir 相同）")
    parser.add_argument("--neg_per_pos", type=int, default=1, help="每筆正樣本對應的負樣本數")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("載入資料...")
    ratings  = load_ratings(args.data_dir)
    id2title = load_movies(args.data_dir)

    print("建構樣本...")
    train_rec, valid_rec, test_rec = build_records(ratings, id2title, args.neg_per_pos)

    for name, records in [("train", train_rec), ("valid", valid_rec), ("test", test_rec)]:
        df = pd.DataFrame(records)
        out_path = os.path.join(args.output_dir, f"{name}.pkl")
        df.to_pickle(out_path)
        pos = (df["label"] == 1).sum()
        neg = (df["label"] == 0).sum()
        print(f"  {name}.pkl: {len(df)} 筆（pos={pos}, neg={neg}）→ {out_path}")

    print("完成。")


if __name__ == "__main__":
    main()
