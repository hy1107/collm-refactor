import pandas as pd
from torch.utils.data import Dataset


class RecDataset(Dataset):
    """MovieLens / Amazon 推薦資料集。

    讀取 pickle 格式的 DataFrame，每筆樣本包含：
    - uid:            用戶 ID（int）
    - iid:            目標物品 ID（int）
    - title:          目標物品標題（str）
    - history_iid:    歷史互動物品 ID 列表
    - history_title:  歷史互動物品標題列表
    - label:          正樣本為 1，負樣本為 0

    Args:
        data_path:     pickle 路徑（不含 .pkl 副檔名）
        dataset_type:  "movielens" 或 "amazon"（目前邏輯相同，預留擴展）
    """

    def __init__(self, data_path: str, dataset_type: str = "movielens"):
        df = pd.read_pickle(data_path + ".pkl")
        self.records = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        return {
            "uid": int(row["uid"]),
            "iid": int(row["iid"]),
            "title": str(row["title"]),
            "history_iid": list(row.get("history_iid", [])),
            "history_titles": list(row.get("history_title", [])),
            "label": int(row["label"]),
        }
