#!/bin/bash
set -e

for num in 10 20 30 40 50 60 70 80 90 100
do
  for topk in 50 
  do
  
  ##############################
  # COCO
  ##############################
  echo '********coco*********'
  echo $num
  python demo_step1.py --dataset 'coco' --num_epoch_pretext $num --topk_p $topk | tee -a ./results/coco.log 
  
  ##############################
  # NUS21
  ##############################
  echo '********nus21*********'
  echo $num
  python demo_step1.py --dataset 'nus21' --num_epoch_pretext $num --topk_p $topk | tee -a ./results/nus21.log 
  
  done
done

