#!/usr/bin/env bash

K=10
LIKE_THRESH=5

python evaluation.py \
    --out_folder "eval/knnmodel_K-$K-Like-$LIKE_THRESH" \
    --model knnmodel \
    --K $K \
    --like_threshold $LIKE_THRESH


