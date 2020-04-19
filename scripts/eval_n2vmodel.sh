#!/usr/bin/env bash

K=10
LIKE_THRESH=5

python evaluation.py \
    --out_folder "eval/n2vmodel" \
    --model n2vmodel \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --n2vmodel_embed_fn 'n2v_save/node2vec_dict'
