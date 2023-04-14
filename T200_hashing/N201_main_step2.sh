#!/bin/bash
set -e

for i in 16 32 64 128
do
    num=20
    inter=1
    echo $num
    echo $i

    ##############################
    # NUS21 
    ##############################
    CUDA_VISIBLE_DEVICES=0 python demo_train_step2.py \
    --nbit $i \
    --dataset 'nus21' \
    --num_epoch $num \
    --inter $inter \
    --num_layer 2 \
    --botk 1.3 \
    --topk 5 \
    --gama 1 \
    --beta 1 #0.2

    cd matlab &&
    matlab -nojvm -nodesktop -r "demo_eval($i, 'nus21', 'N201_new', $num, $inter); quit;" &&
    cd ..
done

