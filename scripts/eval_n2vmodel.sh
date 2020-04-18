#!/usr/bin/env bash

K=10
LIKE_THRESH=5

python eval_model.py \
    --out_folder "eval/n2vmodel_debug" \
    --model n2vmodel \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --n2vmodel_embed_fn 'node2vec_model' \


