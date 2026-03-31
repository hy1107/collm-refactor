"""
Stage 1 訓練腳本（純文字 LoRA 微調，無協同信號）

直接執行：
    python scripts/run_stage1_train.py

所有設定都在最上面的 CONFIG 區塊，修改後重新執行即可。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from transformers import TrainingArguments

from collm.training.config import CoLLMConfig, BackboneConfig, RecEncoderConfig, DataConfig
from collm.model.backbone import build_backbone
from collm.model.collm import CoLLMModel
from collm.data.dataset import RecDataset
from collm.data.collator import RecDataCollator
from collm.data.prompts import get_prompt, get_answer
from collm.training.trainer import CoLLMTrainer

# ============================================================
# CONFIG：修改這裡的路徑和超參數
# ============================================================

# 資料路徑（資料夾下需有 train.pkl 和 valid.pkl）
DATA_DIR = "C:/Users/haoyu/Desktop/vicuna/data/ml-1m"

# 基礎模型（HuggingFace model id 或本地路徑）
BASE_MODEL = "lmsys/vicuna-7b-v1.5"

# 輸出 checkpoint 路徑
OUTPUT_DIR = "./stage1_fixed"

# 訓練超參數
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
LOGGING_STEPS = 50
SAVE_STEPS = 500

# LoRA 設定
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# 資料設定
MAX_HISTORY_LEN = 20
MAX_SEQ_LEN = 512

# ============================================================


def preview_prompts(data_path: str, n: int = 3):
    """訓練前先印出前 n 筆的 prompt，確認格式正確。"""
    df = pd.read_pickle(data_path + ".pkl")
    print("=" * 60)
    print("Prompt 格式預覽（前 {} 筆）".format(n))
    print("=" * 60)
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        history_titles = list(
            row.get("history_titles", row.get("history_title", row.get("his_title", [])))
        )
        # 過濾空字串
        history_titles = [t for t in history_titles if t]
        target_title = str(row["title"])
        label = int(row["label"])

        prompt = get_prompt(
            history_titles=history_titles,
            target_title=target_title,
            use_collaborative_signal=False,
            max_history=MAX_HISTORY_LEN,
        )
        answer = get_answer(label)
        print(f"--- 樣本 {i+1}  uid={row['uid']}  label={label} ---")
        print(prompt + answer)
        print()


def main():
    # ----------------------------------------------------------
    # 1. 載入資料，確認 user/item 數量
    # ----------------------------------------------------------
    train_pkl = DATA_DIR + "/train"
    valid_pkl = DATA_DIR + "/valid"

    train_df = pd.read_pickle(train_pkl + ".pkl")
    user_num = int(train_df["uid"].max()) + 1
    item_num = int(train_df["iid"].max()) + 1
    print(f"[資料] train 筆數={len(train_df)}, user_num={user_num}, item_num={item_num}")
    print(f"[資料] 欄位：{train_df.columns.tolist()}")
    pos_ratio = train_df["label"].mean()
    print(f"[資料] 正樣本比例：{pos_ratio:.2%}  負樣本比例：{1-pos_ratio:.2%}")

    # ----------------------------------------------------------
    # 2. 印出 prompt 格式讓你確認
    # ----------------------------------------------------------
    preview_prompts(train_pkl)

    # ----------------------------------------------------------
    # 3. 建立 config
    # ----------------------------------------------------------
    cfg = CoLLMConfig(
        rec=RecEncoderConfig(
            encoder_type="sasrec",
            checkpoint_path="",
            embedding_dim=64,
            use_collaborative_signal=False,
            user_num=user_num,
            item_num=item_num,
            max_seq_len=50,
            n_layers=2,
            n_heads=2,
            dropout=0.2,
            n_gcn_layers=3,
        ),
        backbone=BackboneConfig(
            model_name_or_path=BASE_MODEL,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_target_modules=LORA_TARGET_MODULES,
            load_in_8bit=False,
        ),
        data=DataConfig(
            data_path=DATA_DIR,
            dataset_type="movielens",
            max_history_len=MAX_HISTORY_LEN,
            max_seq_len=MAX_SEQ_LEN,
        ),
    )

    # ----------------------------------------------------------
    # 4. 載入 backbone + tokenizer，驗證 Yes/No token
    # ----------------------------------------------------------
    backbone, tokenizer = build_backbone(cfg.backbone)

    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens  = tokenizer.encode(" No",  add_special_tokens=False)
    print(f"\n[tokenizer]  ' Yes' → token IDs: {yes_tokens}  decoded: {[tokenizer.decode([t]) for t in yes_tokens]}")
    print(f"[tokenizer]  ' No'  → token IDs: {no_tokens}  decoded: {[tokenizer.decode([t]) for t in no_tokens]}")
    if len(yes_tokens) != 1 or len(no_tokens) != 1:
        print("[WARNING] Yes 或 No 被 tokenize 成多個 token，請確認 tokenizer 設定！")

    yes_id = yes_tokens[0]
    no_id  = no_tokens[0]

    # ----------------------------------------------------------
    # 5. 建立 model
    # ----------------------------------------------------------
    model = CoLLMModel(backbone, tokenizer, config=cfg)

    # ----------------------------------------------------------
    # 6. 建立 dataset + collator
    # ----------------------------------------------------------
    train_ds = RecDataset(train_pkl, cfg.data.dataset_type)
    eval_ds  = RecDataset(valid_pkl, cfg.data.dataset_type)
    print(f"\n[dataset] train={len(train_ds)} 筆, valid={len(eval_ds)} 筆")

    collator = RecDataCollator(
        tokenizer=tokenizer,
        use_collaborative_signal=False,
        max_seq_len=MAX_SEQ_LEN,
        max_history_len=MAX_HISTORY_LEN,
        is_sequential=False,   # Stage 1 不需要序列歷史
    )

    # ----------------------------------------------------------
    # 7. 訓練
    # ----------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
    )

    trainer = CoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        yes_token_id=yes_id,
        no_token_id=no_id,
    )

    print("\n[training] 開始 Stage 1 訓練...")
    trainer.train()

    # ----------------------------------------------------------
    # 8. 存 checkpoint
    # ----------------------------------------------------------
    model.backbone.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[done] Stage 1 checkpoint 已存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
