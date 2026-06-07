#!/bin/bash
PROJECT_DIR="/home/dengyunhui/repo/VAD/NEOVAD/src"
cd ${PROJECT_DIR}

GPUS=(0 1 2 3 4 5 6)   # 7 GPUs
NUM_GPUS=${#GPUS[@]}

index=0

for w in 0.05 0.1 0.2 0.3 0.4 0.5; do
    GPU=${GPUS[$((index % NUM_GPUS))]}
    echo "启动任务：text_adapt_weight=${w}  -> GPU ${GPU}"

    python main.py \
        --mode train \
        --dataset ucf \
        --test "text_weight_${w}" \
        --device "cuda:${GPU}" \
        --lamda2 1 \
        --lamda3 2 \
        --text_adapt_weight ${w} &

    index=$((index + 1))
done

wait
echo "所有任务已启动完成！"
