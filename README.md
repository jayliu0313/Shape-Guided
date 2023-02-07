# Shape-Guided: Shape-Guided Dual-Memory Learning for 3D Anomaly Detection



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

### Download Dataset & Preprocessing
```
mkdir MVTec3DAD && cd MVTec3DAD

```

It will take few minutes to remove the backgoround of the point cloud, and point cloud will be divided into multiple grid for each instance. 
### Preprocessing
```
python tools/preprocessing.py DATASET_PATH
python cut_grid.py --save_grid_path GRID_PATH
```

There is the best_ckpt in ./checkpoint, or you can train the model by yourself.
### Train Our 3D Expert Model
```
python pretrain.py --grid_path GRID_PATH
```

### Train memory and Testing
```
python train_memory.py --datasets_path DATASET_PATH --grid_path GRID_PATH
```

## Reference
Our memory architecture is refer to https://github.com/eliahuhorwitz/3D-ADS  
3D expert model is modified from https://github.com/mabaorui/PredictableContextPrior



