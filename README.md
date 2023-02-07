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

It will take few minutes to remove the backgoround of the point cloud. 
### preprocessing
```
python tools/preprocessing.py DATASET_PATH
python cut_grid.py
```

### training 3D expert model
```
```
