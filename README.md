# Room Segmentation
## Environment
* The code has been tested on Linux with python 3.8, torch 1.9.0, and cuda 11.1.
  * Install pytorch and other required packages:
  ```shell
  # adjust the cuda version accordingly
  pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirements.txt
  ```
  * Compile the deformable-attention modules (from [deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)) and the differentiable rasterization module (from [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer)):
  ```shell
  cd models/ops
  sh make.sh

  # unit test for deformable-attention modules (should see all checking is True)
  # python test.py

  cd ../../diff_ras
  python setup.py build develop
  ```


## Data

Please download the data from [this link](https://drive.google.com/file/d/1Q9jov3eMGQfzJiWprVm19mQ0Fcq0iFa1/view?usp=drive_link).And put it in [RoomSegmentation/Data/]()

## Checkpoints

Please download and extract the checkpoints of our model from [this link](https://drive.google.com/file/d/1fjxTSX1sb6fJtQ1w8--O4bpgW5lM4xTn/view?usp=drive_link).And put it in [RoomSegmentation/Weights/]()


## Evaluation
The predicted images will be saved in [RoomSegmentation/Results/]() and the AP50 and mIOU will be shown in terminal.
```shell
sudo chmod +x scripts/eval_mp3d.sh
sudo chmod +x scripts/eval_mp3d_enhanced.sh
./scripts/eval_mp3d.sh
./scripts/eval_mp3d_enhanced.sh
```

## Single scene demo
The input is a PLY file (with the Y-axis facing up by default, and preferably with a mesh). The output consists of the segmented room point clouds and their corresponding information. The input and output paths can be modified in the shell (.sh) file. Currently, the default is set to the point cloud of the first floor of GGBL in University of Michigan.
```shell
sudo chmod +x scripts/demo.sh
./scripts/demo.sh
```
