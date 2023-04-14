#!/bin/bash
set -e

for i in 16 32 64 128
do
    num=40
    inter=2
    echo $num
    echo $i

    ##############################
    # COCO
    ##############################
    CUDA_VISIBLE_DEVICES=1 python demo_train_step2.py \
    --nbit $i \
    --dataset 'coco' \
    --num_epoch $num \
    --inter $inter \
    --num_layer 4 \
    --botk 1.3 \
    --topk 30 
   
    cd matlab &&
    matlab -nojvm -nodesktop -r "demo_eval($i, 'coco', 'C201', $num, $inter); quit;" &&
    cd ..
    
done
