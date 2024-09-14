#!/usr/bin/env bash

python main.py --dataset_name=scenecad \
               --dataset_root=data_preprocess/coco_mp3d_implicit\
               --num_queries=800 \
               --num_polys=20 \
               --batch_size=1\
               --semantic_classes=-1 \
               --job_name=train_stru3d
