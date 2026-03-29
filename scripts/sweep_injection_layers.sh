#!/usr/bin/env bash
# sweep_injection_layers.sh
# 對不同 injection_layer 跑 Stage 2，結果存到各自的子目錄。
#
# 用法：
#   bash scripts/sweep_injection_layers.sh \
#       --config    configs/stage2_movielens.yaml \
#       --stage1    /checkpoints/stage1 \
#       --base_dir  /checkpoints/sweep_injection \
#       --layers    "0 3 10 16 20 23"
#
# 選填：
#   --epochs    (預設 2)
#   --bs        per_device_train_batch_size（預設 4）
#   --lr        learning_rate（預設 1e-4）

set -e

# ── 預設值 ──────────────────────────────────────────────────
CONFIG="configs/stage2_movielens.yaml"
STAGE1_CKPT="/checkpoints/stage1"
BASE_DIR="/checkpoints/sweep_injection"
LAYERS="0 3 10 16 20 23"
EPOCHS=2
BS=4
LR=1e-4

# ── 解析 CLI 參數 ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2";      shift 2 ;;
        --stage1)   STAGE1_CKPT="$2"; shift 2 ;;
        --base_dir) BASE_DIR="$2";    shift 2 ;;
        --layers)   LAYERS="$2";      shift 2 ;;
        --epochs)   EPOCHS="$2";      shift 2 ;;
        --bs)       BS="$2";          shift 2 ;;
        --lr)       LR="$2";          shift 2 ;;
        *) echo "未知參數: $1"; exit 1 ;;
    esac
done

mkdir -p "$BASE_DIR"
SUMMARY="$BASE_DIR/results_summary.tsv"
echo -e "injection_layer\toutput_dir" > "$SUMMARY"

echo "======================================"
echo "  Injection Layer Sweep"
echo "  Config       : $CONFIG"
echo "  Stage1 ckpt  : $STAGE1_CKPT"
echo "  Output base  : $BASE_DIR"
echo "  Layers       : $LAYERS"
echo "  Epochs/BS/LR : $EPOCHS / $BS / $LR"
echo "======================================"

for LAYER in $LAYERS; do
    OUT_DIR="$BASE_DIR/layer_${LAYER}"
    echo ""
    echo ">>> injection_layer=$LAYER  →  $OUT_DIR"

    python scripts/train_stage2.py \
        --config                      "$CONFIG" \
        --stage1_checkpoint           "$STAGE1_CKPT" \
        --output_dir                  "$OUT_DIR" \
        --injection_layer             "$LAYER" \
        --num_train_epochs            "$EPOCHS" \
        --per_device_train_batch_size "$BS" \
        --learning_rate               "$LR"

    echo -e "$LAYER\t$OUT_DIR" >> "$SUMMARY"
    echo "<<< layer $LAYER 完成"
done

echo ""
echo "======================================"
echo "  全部完成，結果總覽："
cat "$SUMMARY"
echo "======================================"
