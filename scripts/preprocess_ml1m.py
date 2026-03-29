"""
ML-1M 資料預處理腳本。

產生 train.pkl / valid.pkl / test.pkl，格式與 RecDataset 相容。

切分策略（對應原始 CoLLM 論文）：
    每位用戶按時間排序後：
        test  = 最後一筆正向互動（目標物品必須是暖物品）
        valid = 倒數第二筆（目標物品必須是暖物品）
        train = 倒數第三筆（= 訓練期最後一筆），使用 train_neg_per_pos 個負樣本

    暖物品定義：出現在訓練集正樣本目標中的物品。
    只有目標物品是暖物品的 valid/test 樣本才會被保留（對應論文的暖啟動評估）。

    訓練集每位 user 只有 1 個正樣本（對應論文 ~33K 筆的規模）。

每筆樣本欄位：
    uid, iid, title, history_iid, history_titles, label

用法：
    python scripts/preprocess_ml1m.py \\
        --data_dir   /data/ml-1m \\
        --output_dir /data/ml-1m \\
        --train_neg  5 \\
        --eval_neg   1 \\
        --min_inter  5
"""
import argparse
import os
import random
import pandas as pd
import numpy as np


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


def make_record(uid, iid, title, history, history_titles, label):
    return {
        "uid": uid, "iid": iid, "title": title,
        "history_iid": history[:],
        "history_titles": history_titles[:],
        "label": label,
    }


def build_records(
    ratings: pd.DataFrame,
    id2title: dict,
    train_neg: int,
    eval_neg: int,
    min_inter: int,
) -> tuple:
    # 只保留正向互動（rating > 3），按時間排序
    pos = ratings[ratings["Rating"] > 3].sort_values("Timestamp")

    # 重新編碼為 0-indexed
    user_ids = sorted(pos["UserID"].unique())
    item_ids = sorted(pos["MovieID"].unique())
    u2idx = {u: i for i, u in enumerate(user_ids)}
    i2idx = {it: i for i, it in enumerate(item_ids)}

    # 每個 user 的正向互動序列（時間升序，去重保留首次）
    user_history: dict[int, list] = {}
    for _, row in pos.iterrows():
        uid = u2idx[row["UserID"]]
        iid = i2idx[row["MovieID"]]
        user_history.setdefault(uid, [])
        if iid not in user_history[uid]:
            user_history[uid].append(iid)

    # 過濾互動數不足的用戶
    user_history = {u: h for u, h in user_history.items() if len(h) >= min_inter}

    all_items = list(range(len(item_ids)))

    # 訓練集正樣本目標集合（暖物品集合）
    train_warm_items = set()
    for hist in user_history.values():
        if len(hist) >= 3:
            train_warm_items.add(hist[-3])  # 訓練正樣本的目標物品

    train_records, valid_records, test_records = [], [], []

    for uid, hist in user_history.items():
        titles = [id2title.get(item_ids[iid], "Unknown") for iid in hist]
        neg_pool = list(set(all_items) - set(hist))

        # ── 訓練樣本：倒數第三筆，history = 前面所有互動 ──────────────
        train_target_iid   = hist[-3]
        train_history      = hist[:-3]
        train_target_title = titles[-3]
        train_history_titles = titles[:-3]

        train_records.append(make_record(
            uid, train_target_iid, train_target_title,
            train_history, train_history_titles, 1
        ))
        for _ in range(train_neg):
            neg = random.choice(neg_pool)
            train_records.append(make_record(
                uid, neg, id2title.get(item_ids[neg], "Unknown"),
                train_history, train_history_titles, 0
            ))

        # ── Valid 樣本：倒數第二筆，只有暖物品才保留 ──────────────────
        valid_target_iid = hist[-2]
        if valid_target_iid in train_warm_items:
            valid_history        = hist[:-2]
            valid_target_title   = titles[-2]
            valid_history_titles = titles[:-2]
            valid_records.append(make_record(
                uid, valid_target_iid, valid_target_title,
                valid_history, valid_history_titles, 1
            ))
            for _ in range(eval_neg):
                neg = random.choice(neg_pool)
                valid_records.append(make_record(
                    uid, neg, id2title.get(item_ids[neg], "Unknown"),
                    valid_history, valid_history_titles, 0
                ))

        # ── Test 樣本：最後一筆，只有暖物品才保留 ─────────────────────
        test_target_iid = hist[-1]
        if test_target_iid in train_warm_items:
            test_history        = hist[:-1]
            test_target_title   = titles[-1]
            test_history_titles = titles[:-1]
            test_records.append(make_record(
                uid, test_target_iid, test_target_title,
                test_history, test_history_titles, 1
            ))
            for _ in range(eval_neg):
                neg = random.choice(neg_pool)
                test_records.append(make_record(
                    uid, neg, id2title.get(item_ids[neg], "Unknown"),
                    test_history, test_history_titles, 0
                ))

    return train_records, valid_records, test_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_neg",  type=int, default=5,
                        help="訓練集每筆正樣本對應的負樣本數（預設 5）")
    parser.add_argument("--eval_neg",   type=int, default=1,
                        help="valid/test 每筆正樣本對應的負樣本數（預設 1）")
    parser.add_argument("--min_inter",  type=int, default=5,
                        help="用戶最少正向互動數（預設 5）")
    parser.add_argument("--seed",       type=int, default=42)
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
