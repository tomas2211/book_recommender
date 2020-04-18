#!/usr/bin/env bash

K=10
LIKE_THRESH=5

python eval_model.py \
    --out_folder "eval/pmodel_K-$K-Like-$LIKE_THRESH-sig1-0_inclimplicit" \
    --model pmodel \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --pmodel_implicit_is_like \
    --pmodel_sigmul 1.0

python eval_model.py \
    --out_folder "eval/pmodel_K-$K-Like-$LIKE_THRESH-sig0-1_inclimplicit" \
    --model pmodel \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --pmodel_implicit_is_like \
    --pmodel_sigmul 0.1

python eval_model.py \
    --out_folder "eval/pmodel_K-$K-Like-$LIKE_THRESH-sig1-0" \
    --model pmodel \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --pmodel_sigmul 1.0

python eval_model.py \
    --out_folder "eval/pmodel_K-$K-Like-$LIKE_THRESH-sig0-1" \
    --model pmodel \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --pmodel_sigmul 0.1