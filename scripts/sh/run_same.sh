#!/bin/bash

NUM_GPUS=7
RUNS=7
PROJECT_DIR="/home/dengyunhui/repo/VAD/PLOVAD/src"
cd ${PROJECT_DIR}

for i in $(seq 0 $((RUNS-1)))
do
    GPU=$((i % NUM_GPUS))
    TEST_NAME="test_${i}"

    echo "Launching job ${i} on GPU ${GPU}"

    python main.py \
        --mode train \
        --dataset ucf \
        --test ${TEST_NAME} \
        --device cuda:${GPU} \
        --lamda2 1 \
        --lamda3 2 &
done

wait
echo "All jobs finished!"
