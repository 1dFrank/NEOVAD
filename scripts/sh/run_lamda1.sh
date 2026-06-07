#!/bin/bash
PROJECT_DIR="/home/dengyunhui/repo/VAD/NEOVAD/src"
cd ${PROJECT_DIR}

GPUS=(0 1 2 3 4 5 6)   # 7 GPUs
NUM_GPUS=${#GPUS[@]}

index=0

# 配置 lamda2 的范围
START=0.2
END=1.4
STEP=0.2

# 生成浮点数循环
w=$START
while (( $(echo "$w <= $END" | bc -l) )); do
    GPU=${GPUS[$((index % NUM_GPUS))]}

    echo "启动任务：lamda2=${w}  -> GPU ${GPU}"

    python main.py \
        --mode train \
        --dataset ucf \
        --test "lamda2_${w}" \
        --device "cuda:${GPU}" \
        --lamda2 ${w} \
        --lamda3 2 &

    index=$((index + 1))
    w=$(echo "$w + $STEP" | bc)
done

wait
echo "所有任务已启动完成！"
