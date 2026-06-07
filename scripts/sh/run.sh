#!/bin/bash

# 启动多个实验并行运行

# 数据集参数
PROJECT_DIR="/home/dengyunhui/repo/VAD/PLOVAD/src"
DATASET="ucf"
MODE="train"

# GPU 数量
NUM_GPUS=7

cd ${PROJECT_DIR}

# lamda2 从 1 到 10
for LAMDA2 in {18..22}
do
    # 计算分配的 GPU ID（循环使用 0~6）
    GPU_ID=$(( (LAMDA2 - 1) % NUM_GPUS ))

    # test 名称
    TEST_NAME="test_lamda2_${LAMDA2}"

    echo "🚀 Running experiment with lamda2=${LAMDA2} on cuda:${GPU_ID}"

    # 启动训练进程（后台执行 & 输出重定向到日志）
    nohup python main.py \
        --mode ${MODE} \
        --dataset ${DATASET} \
        --test ${TEST_NAME} \
        --device cuda:${GPU_ID} \
        --lamda2 ${LAMDA2} &
done

echo "All experiments started in background."
