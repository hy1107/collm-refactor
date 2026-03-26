import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
from collm.data.prompts import get_prompt, get_answer
from collm.constants import USER_TOKEN, ITEM_TOKEN


class RecDataCollator:
    """將一批 RecDataset 樣本組裝為 CoLLMModel.forward() 所需格式。

    負責：
    - 根據 use_collaborative_signal 選擇 prompt 模板
    - Tokenize prompt + answer，設定 labels（-100 遮蔽 prompt 部分）
    - 定位 [USER_TOKEN] 和 [ITEM_TOKEN] 的 token 位置（-1 表示不存在）
    - 序列模型：pad 歷史 ID 序列為 (batch, max_history_len) 的 LongTensor

    Args:
        tokenizer:                  已注冊占位符 token 的 tokenizer
        use_collaborative_signal:   False → 消融模式（無占位符）
        max_seq_len:                prompt + answer 的最大 token 長度
        max_history_len:            歷史序列的最大長度
        is_sequential:              True → 包含 seq_history（SASRec/DIN）
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        use_collaborative_signal: bool,
        max_seq_len: int,
        max_history_len: int,
        is_sequential: bool,
    ):
        self.tokenizer = tokenizer
        self.use_collaborative_signal = use_collaborative_signal
        self.max_seq_len = max_seq_len
        self.max_history_len = max_history_len
        self.is_sequential = is_sequential
        self._user_tok_id = tokenizer.convert_tokens_to_ids(USER_TOKEN)
        self._item_tok_id = tokenizer.convert_tokens_to_ids(ITEM_TOKEN)

    def __call__(self, samples: list) -> dict:
        input_ids_list, attention_mask_list, labels_list = [], [], []
        user_ids, target_item_ids = [], []
        user_ph_pos, item_ph_pos = [], []
        seq_history_list = []

        for s in samples:
            prompt = get_prompt(
                history_titles=s["history_titles"],
                target_title=s["title"],
                use_collaborative_signal=self.use_collaborative_signal,
                max_history=self.max_history_len,
            )
            answer = get_answer(s["label"])
            full_text = prompt + answer

            enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            ids = enc["input_ids"][0]
            mask = enc["attention_mask"][0]

            prompt_enc = self.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
            prompt_len = prompt_enc["input_ids"].shape[1]

            lbl = ids.clone()
            lbl[:prompt_len] = -100
            lbl[mask == 0] = -100

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(lbl)
            user_ids.append(s["uid"])
            target_item_ids.append(s["iid"])

            user_ph_pos.append(self._find_token_pos(ids, self._user_tok_id))
            item_ph_pos.append(self._find_token_pos(ids, self._item_tok_id))

            if self.is_sequential:
                hist = s["history_iid"][-self.max_history_len:]
                padded = [0] * (self.max_history_len - len(hist)) + hist
                seq_history_list.append(padded)

        batch = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
            "user_ids": torch.tensor(user_ids, dtype=torch.long),
            "target_item_ids": torch.tensor(target_item_ids, dtype=torch.long),
            "user_placeholder_pos": torch.tensor(user_ph_pos, dtype=torch.long),
            "item_placeholder_pos": torch.tensor(item_ph_pos, dtype=torch.long),
        }
        if self.is_sequential:
            batch["seq_history"] = torch.tensor(seq_history_list, dtype=torch.long)

        return batch

    def _find_token_pos(self, ids: Tensor, token_id: int) -> int:
        positions = (ids == token_id).nonzero(as_tuple=True)[0]
        return positions[0].item() if len(positions) > 0 else -1
