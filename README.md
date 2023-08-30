# Shape-Guided Dual-Memory Learning for 3D Anomaly Detection (ICML2023)
### [Paper Link](https://openreview.net/pdf?id=IkSGn9fcPz)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shape-guided-shape-guided-dual-memory/3d-anomaly-detection-and-segmentation-on)](https://paperswithcode.com/sota/3d-anomaly-detection-and-segmentation-on?p=shape-guided-shape-guided-dual-memory)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shape-guided-shape-guided-dual-memory/rgb-3d-anomaly-detection-and-segmentation-on)](https://paperswithcode.com/sota/rgb-3d-anomaly-detection-and-segmentation-on?p=shape-guided-shape-guided-dual-memory)
## Qualitative Result
![image](https://github.com/jayliu0313/Shape-Guided/blob/main/img/complementary_heatmap.png)
Signed Distance Function(SDF) means the method we estimate the point cloud to detect anomaly. <br/>
We utilize the information of the RGB and the corresponding 3D point cloud to detect anomaly and complement each other to get the final score map. <br/>
![image](https://github.com/jayliu0313/Shape-Guided/blob/main/img/Comparison_heatmap.png)
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

### Preprocessing
It will take few minutes to remove the backgoround of the point cloud.
```
python tools/preprocessing.py DATASET_PATH
```
Divided the point cloud into multiple local patches for each instance.<br/>
```
# This is just for building memory and inference.
python cut_patches.py --datasets_path DATASET_PATH --save_grid_path GRID_PATH
```
*Make sure the order of execution of preprocessing.py is before cut_patches.py.* <br/>

### Train Our 3D Expert Model
There is the best checkpoint of the 3D expert model in ```checkpoint/best_ckpt/ckpt_000601.pth```, and you can skip this step. Alternatively, you can train the 3D expert model on your own. So, you need to execute the following commands to get the required training patches which are contained the noise points.<br/>
*Recommend setting the ```save_grid_path``` in the same directory as above.*
```
# This is for training 3D expert model.
python cut_patches.py --datasets_path DATASET_PATH --save_grid_path GRID_PATH --pretrain
```
then,
```
python train_3Dmodel.py --grid_path GRID_PATH --ckpt_path "./checkpoint"
```

### Buid Memory and Inference
The result will be stored in the output directory.
```
python main.py --datasets_path DATASET_PATH --grid_path GRID_PATH --ckpt_path "checkpoint/best_ckpt/ckpt_000601.pth" --viz
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
