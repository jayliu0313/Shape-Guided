import os
import glob
import math
import torch
import argparse
import numpy as np
import tifffile as tiff
from plyfile import PlyData
from tqdm import tqdm

LIMIT = 10000

def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img

def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0)
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0)
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).numpy()

def orgpc_to_unorgpc(organized_pc):
    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    pc_no_zeros = unorganized_pc[nonzero_indices, :]
    return pc_no_zeros, nonzero_indices

def getPointCloud(tiff_path, img_size):
    organized_pc = read_tiff_organized_pc(tiff_path)
    num = 0

    resized_organized_pc = resize_organized_pc(organized_pc, target_height=img_size, target_width=img_size)
    pc, nonzero_idx = orgpc_to_unorgpc(resized_organized_pc)
    pc = np.asarray(pc)
    return pc , nonzero_idx

DATASETS_PATH = "/mnt/home_6T/public/samchu0218/Datasets/mvtec3d_preprocessing/"
classes = [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
        ]


for class_name in classes:
 
    MaxPoint = 0
    MinPoint = 1000000
    MaxSize = 0
    MinSize = 800
    total = 0
    dir_path=[]

    img_path = os.path.join(DATASETS_PATH, class_name, "train")
    dir_path = glob.glob(os.path.join(img_path, 'good', 'xyz') + "/*.tiff")

    img_path = os.path.join(DATASETS_PATH, class_name, "test")
    defect_types = os.listdir(img_path)
    
    for defect_type in defect_types:
        tiff_path = glob.glob(os.path.join(img_path, defect_type, 'xyz') + "/*.tiff")
        dir_path.extend(tiff_path)
    dir_path.sort()

    tiff_size = 224
    for i in tqdm (range(len(dir_path)), desc=f'Class {class_name}'):

        tiff_path = dir_path[i]
        input, nonzero_idx = getPointCloud(tiff_path, tiff_size)
        pointcloud_s = input.astype(np.float32)
        
        #print("tiff size:",tiff_size)
        #print('### pointcloud sparse:', pointcloud_s.shape[0])
        #print('tiff fiel:',tiff_path)
        if pointcloud_s.shape[0] > MaxPoint:
            MaxPoint = pointcloud_s.shape[0]
        if pointcloud_s.shape[0] < MinPoint:
            MinPoint = pointcloud_s.shape[0]

    print('================== Class:',class_name, ' =========================')
    print('Max Point num:',MaxPoint)
    print('Min Point num:',MinPoint)
    #print('Average Point num:', total*1.0 / len(dir_path))
    print('=========================================================')

    