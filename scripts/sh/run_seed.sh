#!/bin/bash
PROJECT_DIR="/home/dengyunhui/repo/VAD/PLOVAD/src"
cd ${PROJECT_DIR}

GPUS=(0 1 2 3 4 5 6)
NUM_GPUS=${#GPUS[@]}

START=2012
END=2024

index=0

for SEED in $(seq $START $END); do
    GPU=${GPUS[$((index % NUM_GPUS))]}
    echo "启动任务：seed=${SEED}  -> GPU ${GPU}"

    python main.py \
        --mode train \
        --dataset ub \
        --test "bias01_seed${SEED}" \
        --device "cuda:${GPU}" \
        --lamda2 1 \
        --lamda3 2 \
        --seed ${SEED} &

    index=$((index + 1))
done

wait
echo "所有任务启动完成！"
