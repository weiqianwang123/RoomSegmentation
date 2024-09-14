#!/usr/bin/env bash
PYTHONPATH=$PYTHONPATH:./RoomFormer
python RoomFormer/eval.py --dataset_name=scenecad \
               --dataset_root=Data/MP3D\
               --eval_set=test\
               --checkpoint=Weights/checkpoint.pth\
               --output_dir=Results/eval_mp3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 