"""
ML-1M 資料預處理腳本（對應原始 CoLLM 論文的精確切分）。

時間切分（zyang1580/CoLLM dataset/ml-1m/processing_ood.ipynb）：
    月份索引 = (year - min_year)*12 + month - min_month，起點=0
    資料跨度：2000/04 ~ 2003/02（共 35 個月，索引 0-34）

    月 0-13  → 歷史期：只累積 user 歷史，不生訓練樣本
    月 14-23 → Train（10 個月）→ 論文 33,891 筆
    月 24-28 → Valid（ 5 個月）→ 論文 10,401 筆
    月 29-33 → Test （ 5 個月）→ 論文  7,331 筆

Label：rating > 3 → y=1，其餘 → y=0（不隨機採樣負樣本）
History：每個樣本的 context 為「此互動之前的所有正向互動（rating > 3）」

每筆樣本欄位：
    uid, iid, title, history_iid, history_titles, label

用法：
    python scripts/preprocess_ml1m.py \\
        --data_dir   /data/ml-1m \\
        --output_dir /data/ml-1m
"""
import argparse
import os
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


def build_records(ratings: pd.DataFrame, id2title: dict) -> tuple:
    # ── 1. 所有評分，按時間排序 ──────────────────────────────────────
    df = ratings.sort_values("Timestamp").reset_index(drop=True)

    # ── 2. 計算月份索引（對應論文 processing_ood.ipynb）──────────────
    dt = pd.to_datetime(df["Timestamp"], unit="s")
    min_year  = dt.min().year
    min_month = dt.min().month                        # 2000, 4
    df["month_idx"] = (dt.dt.year  - min_year) * 12 + dt.dt.month - min_month

    print(f"  資料時間範圍：{dt.min().date()} ~ {dt.max().date()}")
    print(f"  月份索引範圍：{df['month_idx'].min()} ~ {df['month_idx'].max()}")

    # ── 3. 重新編碼 ID ───────────────────────────────────────────────
    user_ids = sorted(df["UserID"].unique())
    item_ids = sorted(df["MovieID"].unique())
    u2idx = {u: i for i, u in enumerate(user_ids)}
    i2idx = {it: i for i, it in enumerate(item_ids)}
    df["uid"]   = df["UserID"].map(u2idx)
    df["iid"]   = df["MovieID"].map(i2idx)
    df["label"] = (df["Rating"] > 3).astype(int)

    # ── 4. 月份 → split 對應 ────────────────────────────────────────
    TRAIN_MONTHS = set(range(14, 24))   # 14-23
    VALID_MONTHS = set(range(24, 29))   # 24-28
    TEST_MONTHS  = set(range(29, 34))   # 29-33
    # 月 0-13：歷史期，不生樣本

    def get_split(m):
        if m in TRAIN_MONTHS: return "train"
        if m in VALID_MONTHS: return "valid"
        if m in TEST_MONTHS:  return "test"
        return "history"   # 月 0-13：只累積歷史

    df["split"] = df["month_idx"].map(get_split)

    for s in ["history", "train", "valid", "test"]:
        n = (df["split"] == s).sum()
        pos = ((df["split"] == s) & (df["label"] == 1)).sum()
        print(f"  {s:8s}: {n:>7} 筆（pos={pos}, neg={n-pos}）")

    # ── 5. 逐時間順序處理，維護遞增歷史 ─────────────────────────────
    user_hist_iid:   dict[int, list] = defaultdict(list)
    user_hist_title: dict[int, list] = defaultdict(list)
    user_pos_seen:   dict[int, set]  = defaultdict(set)

    train_records, valid_records, test_records = [], [], []

    for _, row in df.iterrows():
        uid   = int(row["uid"])
        iid   = int(row["iid"])
        label = int(row["label"])
        title = id2title.get(row["MovieID"], "Unknown")
        split = row["split"]

        hist_iid   = user_hist_iid[uid]
        hist_title = user_hist_title[uid]

        if split == "train":
            train_records.append(
                make_record(uid, iid, title, hist_iid, hist_title, label)
            )
        elif split == "valid":
            valid_records.append(
                make_record(uid, iid, title, hist_iid, hist_title, label)
            )
        elif split == "test":
            test_records.append(
                make_record(uid, iid, title, hist_iid, hist_title, label)
            )
        # split == "history"：只更新歷史，不生樣本

        # 正向互動才加入歷史
        if label == 1 and iid not in user_pos_seen[uid]:
            user_hist_iid[uid].append(iid)
            user_hist_title[uid].append(title)
            user_pos_seen[uid].add(iid)

    return train_records, valid_records, test_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("載入資料...")
    ratings  = load_ratings(args.data_dir)
    id2title = load_movies(args.data_dir)

    print("建構樣本...")
    train_rec, valid_rec, test_rec = build_records(ratings, id2title)

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
