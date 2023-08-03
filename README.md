# Shape-Guided: Shape-Guided Dual-Memory Learning for 3D Anomaly Detection (ICML2023) [Paper Link](https://openreview.net/pdf?id=IkSGn9fcPz)

![image](https://github.com/jayliu0313/Shape-Guided/blob/main/img/complementary_heatmap.png)
## Installation
### Requirement
Linux (Ubuntu 16.04)  
Python 3.6+  
PyTorch 1.7 or higher  
CUDA 10.2 or higher

### create environment
```
git clone https://github.com/jayliu0313/Shape-Guided.git
cd Shape-Guided
conda create --name myenv python=3.6
conda activate myenv
pip install requirement.txt
```

### MvTec3D-AD Dataset
[Here](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) to download dataset

## Implementation
It will take few minutes to remove the backgoround of the point cloud, and point cloud will be divided into multiple patches for each instance. 
  If you cut patches for pretrain, you need to change the variable "is_pretrained" to true in cut_patch.py.
### Preprocessing
```
python tools/preprocessing.py DATASET_PATH
python cut_patches.py --save_grid_path GRID_PATH
```

There is the best_ckpt in ./checkpoint, or you can train the model by yourself, and you need to cut the pretrain grid in cut_patches.py.
### Train Our 3D Expert Model
```
python train_3Dmodel.py --grid_path GRID_PATH
```

### Buid Memory and Testing
The result will be stored in the output directory.
```
python main.py --datasets_path DATASET_PATH --grid_path GRID_PATH
```

## Reference
Our memory architecture is refer to https://github.com/eliahuhorwitz/3D-ADS  
3D expert model is modified from https://github.com/mabaorui/PredictableContextPrior
