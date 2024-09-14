#!/usr/bin/env bash
PYTHONPATH=$PYTHONPATH:./RoomFormer
python scripts/ply_to_density.py --ply_path=Data/umich_ggbl.ply \
                                 --output_path=Results/demo/density.png
            

python RoomFormer/demo.py --image_path=Results/demo/density.png\
                          --checkpoint=Weights/checkpoint.pth\
                          --output_dir=Results/demo \

python scripts/save_room.py --ply_file_path=Data/umich_ggbl.ply\
                            --json_file_path=Results/demo/room_polygons.json\
                            --output_dir=Results/demo/rooms