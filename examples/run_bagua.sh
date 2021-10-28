#!/bin/bash

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=19001
NNODES=1
NODE_RANK=0
# WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="/share/liuchengjun/megatron_data/my-bert_text_sentence"
VOCAB_FILE="/share/liuchengjun/megatron_data/bert-large-cased-vocab.txt"
CHECKPOINT_PATH="./bert_checkpoint"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m bagua.distributed.launch --set_additional_flag $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --DDP-impl bagua \
       --top-k 2 \
       --fmoefy \
       --num-experts 2 \
       --num-layers 1 \
       --hidden-size 4 \
       --num-attention-heads 2 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 32 \
       --max-position-embeddings 32 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
