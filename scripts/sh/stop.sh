#!/bin/bash

# =========================================
# 停止所有后台运行的 Python 训练进程
# （特别是 main.py 或包含 cuda 参数的任务）
# =========================================

echo "正在查找后台运行的实验进程..."

# 查找所有包含 main.py 的进程
PIDS=$(ps -ef | grep "python main.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "未发现正在运行的实验进程。"
else
    echo "发现以下进程:"
    ps -ef | grep "python main.py" | grep -v grep

    echo "🔪 开始终止这些进程..."
    for PID in $PIDS; do
        kill -9 $PID
        echo "   ➤ 已终止进程 PID: $PID"
    done

    echo "所有后台实验已停止。"
fi
