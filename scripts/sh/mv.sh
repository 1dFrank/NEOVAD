#!/bin/bash

BASE="/data/dengyunhui/all_datasets/UBnormal_features_reorg/train"
NORMAL_DIR="$BASE/normal"
ABNORMAL_DIR="$BASE/abnormal"

echo "修正开始..."
echo "从: $NORMAL_DIR"
echo "移动到: $ABNORMAL_DIR"

# 创建 abnormal 目录（如果不存在）
mkdir -p "$ABNORMAL_DIR"

# 识别并移动误放的 abnormal 文件
for f in "$NORMAL_DIR"/*.npy; do
    fname=$(basename "$f")
    if [[ "$fname" == abnormal_* ]]; then
        echo "发现 abnormal 文件：$fname → 移到 abnormal/"
        mv "$f" "$ABNORMAL_DIR"/
    fi
done

echo "修正完成！"
