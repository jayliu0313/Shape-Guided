# Shape-Guided: Shape-Guided Dual-Memory Learning for 3D Anomaly Detection

## Installation
You can execute the following command.
### git clone
```
git clone https://github.com/jayliu0313/Shape-Guided.git
cd Shape-Guided
```
### create environment
```
conda create --name myenv python=3.6
conda activate myenv
pip install requirement.txt
```
### download dataset & preprocessing
```
mkdir MVTec3DAD && cd MVTec3DAD

```

It will take few minutes to remove the backgoround of the point cloud, and point cloud will be divided into multiple grid for each instance. 
### preprocessing
```
python tools/preprocessing.py DATASET_PATH
python cut_grid.py --save_grid_path GRID_PATH
```
There is the best_ckpt in ./checkpoint, or you can train the model by yourself.
### train 3D expert model
```
python pretrain.py --grid_path GRID_PATH
```

### train memory and implement inference
python train_memory.py --datasets_path DATASET_PATH --grid_path GRID_PATH

## Reference
Our memory architecture is refer to https://github.com/eliahuhorwitz/3D-ADS  
3D expert model is modified from https://github.com/mabaorui/PredictableContextPrior



