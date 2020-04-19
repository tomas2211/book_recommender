#!/usr/bin/env bash

K=10
LIKE_THRESH=5
GRAPH_PATH='data/corat_graph_18-04-2020_1655'

python eval_model.py \
    --out_folder "eval/gmodel_K-$K-Like-$LIKE_THRESH-sig_1-0_mincorat_2" \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --model gmodel \
    --gmodel_graphpath $GRAPH_PATH \
    --gmodel_sigmul 1 \
    --gmodel_minimal_corats 2


python eval_model.py \
    --out_folder "eval/gmodel_K-$K-Like-$LIKE_THRESH-sig_1-0_mincorat_4" \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --model gmodel \
    --gmodel_graphpath $GRAPH_PATH \
    --gmodel_sigmul 1 \
    --gmodel_minimal_corats 4

python eval_model.py \
    --out_folder "eval/gmodel_K-$K-Like-$LIKE_THRESH-sig_1-0_mincorat_8" \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --model gmodel \
    --gmodel_graphpath $GRAPH_PATH \
    --gmodel_sigmul 1 \
    --gmodel_minimal_corats 8



python eval_model.py \
    --out_folder "eval/gmodel_K-$K-Like-$LIKE_THRESH-sig_0-1_mincorat_2" \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --model gmodel \
    --gmodel_graphpath $GRAPH_PATH \
    --gmodel_sigmul 0.1 \
    --gmodel_minimal_corats 2


python eval_model.py \
    --out_folder "eval/gmodel_K-$K-Like-$LIKE_THRESH-sig_0-1_mincorat_4" \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --model gmodel \
    --gmodel_graphpath $GRAPH_PATH \
    --gmodel_sigmul 0.1 \
    --gmodel_minimal_corats 4

python eval_model.py \
    --out_folder "eval/gmodel_K-$K-Like-$LIKE_THRESH-sig_0-1_mincorat_8" \
    --K $K \
    --like_threshold $LIKE_THRESH \
    --model gmodel \
    --gmodel_graphpath $GRAPH_PATH \
    --gmodel_sigmul 0.1 \
    --gmodel_minimal_corats 8