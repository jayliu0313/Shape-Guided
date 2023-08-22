<meta name="google-site-verification" content="RAL5KyGRWjjA5AdzGbDHGexz0ntRLhEj6enu5kdnmDc" />
# Shape-Guided: Shape-Guided Dual-Memory Learning for 3D Anomaly Detection (ICML2023)
### [Paper Link](https://openreview.net/pdf?id=IkSGn9fcPz)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shape-guided-shape-guided-dual-memory/3d-anomaly-detection-and-segmentation-on)](https://paperswithcode.com/sota/3d-anomaly-detection-and-segmentation-on?p=shape-guided-shape-guided-dual-memory)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shape-guided-shape-guided-dual-memory/rgb-3d-anomaly-detection-and-segmentation-on)](https://paperswithcode.com/sota/rgb-3d-anomaly-detection-and-segmentation-on?p=shape-guided-shape-guided-dual-memory)
![image](https://github.com/jayliu0313/Shape-Guided/blob/main/img/complementary_heatmap.png)
We utilize the information of the RGB and the 3D point cloud to detect anomaly and complement each other. <br/>
Here SDF means the method of 3D point cloud.
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
It will take few minutes to remove the backgoround of the point cloud, and point cloud will be divided into multiple patches for each instance. <br/>
If you cut patches for pretrain, you need to change the variable "is_pretrained" to true in cut_patch.py.<br/>
### Preprocessing
```
python tools/preprocessing.py DATASET_PATH
python cut_patches.py --save_grid_path GRID_PATH
```

### Train Our 3D Expert Model
There is the best checkpoint of the 3D expert model in ./checkpoint.<br/>
Or you can train the 3D model by yourself, you need to cut the pretrain 3D grid in cut_patches.py and change the variable "is_pretrained" to true in cut_patch.py.<br/>
```
python train_3Dmodel.py --grid_path GRID_PATH
```

### Buid Memory and Testing
The result will be stored in the output directory.
```
python main.py --datasets_path DATASET_PATH --grid_path GRID_PATH
```

## Citation
If our paper is useful for your research, please cite our paper. Thank you!
```
@InProceedings{pmlr-v202-chu23b,
  title = {Shape-Guided Dual-Memory Learning for 3D Anomaly Detection},
  author = {Chu, Yu-Min and Liu, Chieh and Hsieh, Ting-I and Chen, Hwann-Tzong and Liu, Tyng-Luh},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {6185--6194},
  year = {2023},
}
```

## Reference
Our memory architecture is refer to https://github.com/eliahuhorwitz/3D-ADS  
3D expert model is modified from https://github.com/mabaorui/PredictableContextPrior
