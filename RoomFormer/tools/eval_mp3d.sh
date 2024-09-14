#!/usr/bin/env bash

python eval.py --dataset_name=scenecad \
               --dataset_root=/home/qianwei/RoomFormer-main/data_preprocess/coco_mp3d\
               --eval_set=test\
               --checkpoint=/home/qianwei/RoomFormer-main/output/500_epoch_batch_1_859/checkpoint0859.pth\
               --output_dir=eval_mp3d_stru_better \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 
