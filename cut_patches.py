import os
import glob
import math
import torch
import argparse
import numpy as np
import tifffile as tiff
from plyfile import PlyData

# sample
from scipy.spatial import cKDTree

# fps and knn
from utils.pointnet_util import sample_and_group


IS_PRTRAIN = False
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--point_num', type=int, default=500)  # number of the points in the knn group.
parser.add_argument('--sample_num', type=int, default=20)   # random sample fake points from for pretraining
parser.add_argument('--save_grid_path', type=str, default="save_grid_path") # save here, pretrain ,train, test grid need to save in the same dir.
if IS_PRTRAIN:
    parser.add_argument('--group_mul', type=int, default=5)     # The sample points is n times more than the original points
    parser.add_argument('--datasets_path', type=str, default="dataset_path")
    classes = ["*"]
else:
    parser.add_argument('--group_mul', type=int, default=10)     # The sample points is n times more than the original points
    parser.add_argument('--datasets_path', type=str, default="dataset_path")
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
a = parser.parse_args() 
DATASETS_PATH  = a.datasets_path
SAVE_GRID_PATH = a.save_grid_path
SAMPLE_NUMBER = a.sample_num
GROUP_SIZE = a.point_num    
GROUP_MUL = a.group_mul


def file_process(split, img_path):
    for i in range(len(img_path)):

        if split in ['pretrain']:
            category_dir = img_path[i].split(DATASETS_PATH)[-1]
            category_dir = category_dir.replace(".ply", ".npz")
            category_dir = category_dir.replace(".tiff", ".npz")
            category_dir = category_dir.replace("pretrain_data", "pretrain")
            save_path = SAVE_GRID_PATH + category_dir
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            category_dir = img_path[i].split(DATASETS_PATH)[-1]
            category_dir = category_dir.replace(".tiff", ".npz")
            category_dir = category_dir.replace("xyz", "npz")
            save_path = SAVE_GRID_PATH + category_dir
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        print('\n<Sampling>')
        print('img_path = ',img_path[i])
        split_patch(split, img_path[i], save_path)  # img_path[i] is the input tiff file

def get_data_loader(split, class_name):

    if  split in ['pretrain']:
        dir_path = glob.glob(os.path.join(DATASETS_PATH, "pretrain_data", class_name) + "/*.tiff")
        dir_path.extend(glob.glob(os.path.join(DATASETS_PATH, "pretrain_data") + "/*/*.ply"))
        dir_path.sort()

    elif split in ['train']:
        img_path = os.path.join(DATASETS_PATH, class_name, "train")
        dir_path = glob.glob(os.path.join(img_path, 'good', 'xyz') + "/*.tiff")
        dir_path.sort()

    elif split in ['test']:
        img_path = os.path.join(DATASETS_PATH, class_name, "test")
        defect_types = os.listdir(img_path)
        dir_path=[]
        for defect_type in defect_types:
            tiff_path = glob.glob(os.path.join(img_path, defect_type, 'xyz') + "/*.tiff")
            dir_path.extend(tiff_path)
        dir_path.sort()

    elif split in ['validation']:
        img_path = os.path.join(DATASETS_PATH, class_name, "validation")
        defect_types = os.listdir(img_path)
        dir_path=[]
        for defect_type in defect_types:
            tiff_path = glob.glob(os.path.join(img_path, defect_type, 'xyz') + "/*.tiff")
            dir_path.extend(tiff_path)
        dir_path.sort()

    file_process(split, dir_path)

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
    resized_organized_pc = resize_organized_pc(organized_pc, target_height=img_size, target_width=img_size)
    pc, nonzero_idx = orgpc_to_unorgpc(resized_organized_pc)
    return pc , nonzero_idx


###################################################################

def new_sample_query_points(target_point_clouds, query_num):
    
    sample = []
    x = int(math.ceil(query_num * 0.625))
    y = query_num - x
    pnts = target_point_clouds
    ptree = cKDTree(pnts)
    for i in range(x):
        
        sigmas = []
        for p in np.array_split(pnts,100,axis=0):
            d = ptree.query(p,51)
            sigmas.append(d[0][:,-1])
        
        sigmas = np.concatenate(sigmas)
        tt = pnts + 0.5*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        sample.append(tt)
    
    for i in range(y):
        
        sigmas = []
        for p in np.array_split(pnts,100,axis=0):
            d = ptree.query(p,51)
            sigmas.append(d[0][:,-1])
        
        sigmas = np.concatenate(sigmas)
        tt = pnts + 1.0*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        sample.append(tt)
    
    sample = np.array(sample).reshape(-1,3)
    return sample

def pretrain_normal_points(ps_gt, ps):
    tt =  0
    if((np.max(ps_gt[:,0])-np.min(ps_gt[:,0]))>(np.max(ps_gt[:,1])-np.min(ps_gt[:,1]))):
        tt = (np.max(ps_gt[:,0])-np.min(ps_gt[:,0]))
    else:
        tt = (np.max(ps_gt[:,1])-np.min(ps_gt[:,1]))
    if(tt < (np.max(ps_gt[:,2])-np.min(ps_gt[:,2]))):
        tt = (np.max(ps_gt[:,2])-np.min(ps_gt[:,2]))

    tt = 10/(10*tt)
    ps_gt = ps_gt*tt
    ps = ps*tt

    t = np.mean(ps_gt,axis = 0)
    ps_gt = ps_gt - t
    ps = ps - t
    return ps_gt, ps, (t, tt)

def find_NN(points , samples):
    
    points = points.reshape(-1,3)
    points_KDtree = cKDTree(points)
    _ , vertex_ids = points_KDtree.query(samples, p=2, k = 1)
    vertex_ids = np.asarray(vertex_ids)
    gt_for_NN = points[vertex_ids].reshape(-1,3)
    return gt_for_NN

def sample_query_points(target_point_clouds, query_num):

    sample = []
    sample_near = []
    gt_kd_tree = cKDTree(target_point_clouds)
    x = int(math.ceil(query_num * 0.625))
    y = query_num - x

    for i in range(x):
        
        pnts = target_point_clouds
        ptree = cKDTree(pnts)
        sigmas = []
        for p in np.array_split(pnts,100,axis=0):
            d = ptree.query(p,51)
            sigmas.append(d[0][:,-1])
        
        sigmas = np.concatenate(sigmas)
        tt = pnts + 0.5*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        sample.append(tt)
        _ , vertex_ids = gt_kd_tree.query(tt, p=2, k = 1)
    
        vertex_ids = np.asarray(vertex_ids)
        sample_near.append(target_point_clouds[vertex_ids].reshape(-1,3))
    
    for i in range(y):
        
        pnts = target_point_clouds
        ptree = cKDTree(pnts)
        sigmas = []
        for p in np.array_split(pnts,100,axis=0):
            d = ptree.query(p,51)
            sigmas.append(d[0][:,-1])
        
        sigmas = np.concatenate(sigmas)
        tt = pnts + 1.0*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        sample.append(tt)
        _ , vertex_ids = gt_kd_tree.query(tt, p=2, k = 1)

        vertex_ids = np.asarray(vertex_ids)
        sample_near.append(target_point_clouds[vertex_ids].reshape(-1,3))
        
    sample = np.asarray(sample).reshape(-1,3)
    sample_near = np.asarray(sample_near).reshape(-1,3)
    return sample, sample_near

def split_patch(split, input_file, save_path):
    points = []
    if('.ply' in input_file):
        data = PlyData.read(input_file)
        v = data['vertex'].data
        v = np.asarray(v)
        for i in range(v.shape[0]):
            points.append(np.array([v[i][0],v[i][1],v[i][2]]))
        points = np.asarray(points)

    elif('.tiff' in input_file):
        input, nonzero_idx = getPointCloud(input_file, 224)
        points = np.asarray(input)

    pointcloud_s = points.astype(np.float32)
    print('### pointcloud sparse:', pointcloud_s.shape[0])

    pointcloud_s_t = pointcloud_s - np.array([np.min(pointcloud_s[:,0]),np.min(pointcloud_s[:,1]),np.min(pointcloud_s[:,2])])
    pointcloud_s_t = pointcloud_s_t / (np.array([np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0])]))
    pointcloud_s = pointcloud_s_t
    
    num_group = round(pointcloud_s.shape[0] * GROUP_MUL / GROUP_SIZE)   # num_group:number of knn groups.

    # FPS + KNN Patch Cutting
    points_gt_all = np.expand_dims(pointcloud_s, axis=0)
    points_gt_all = torch.tensor(points_gt_all)
    org_idx_table = nonzero_idx

    grouped_xyz, org_idx_all = sample_and_group(xyz=points_gt_all, npoint=num_group, nsample=GROUP_SIZE, org_idx_table=org_idx_table)
    grouped_xyz = np.squeeze(grouped_xyz.cpu().data.numpy())

    org_idx_all = np.squeeze(org_idx_all)
    re_org_idx_all = org_idx_all.reshape(org_idx_all.shape[0] * org_idx_all.shape[1])
    no_dup_idx_all = np.unique(re_org_idx_all, axis=0)

    print("Test whether the point cloud is covered")
    print("group_size:", GROUP_SIZE, "num_group:", num_group)
    print("points_gt_all", points_gt_all.shape)
    print("no_dup_idx_all", no_dup_idx_all.shape)
    print('#')

    if split in ['pretrain']:
        origin_all = []
        sample_all = []
        sample_near_all = []
        trans_all = []
        
        for patch in range(grouped_xyz.shape[0]):

            samples = new_sample_query_points(grouped_xyz[patch], SAMPLE_NUMBER)    # sample query point
            points, samples, trans  = pretrain_normal_points(grouped_xyz[patch], samples)
            sample_near = find_NN(points, samples)

            origin_all.append(grouped_xyz[patch])
            sample_all.append(samples)
            sample_near_all.append(sample_near)
            trans_all.append(trans)

        print("group_size:", GROUP_SIZE, "num_group:", num_group)
        print("grouped_xyz", grouped_xyz.shape)
        print("----------------------------------------")
        np.savez(save_path, origins_all=origin_all, samples_all=sample_all, points_all=sample_near_all)
    else:
        np.savez(save_path, points_gt=grouped_xyz, points_idx=org_idx_all)



if __name__ == '__main__':
    print(torch.__version__)
    print("Start to get patch")
    for cls in classes:
        print('Class:',cls)
        if IS_PRTRAIN:
            get_data_loader('pretrain', cls)
        else:
            get_data_loader('train', cls)
            get_data_loader('test', cls)
            get_data_loader('validation', cls)
    print('Finish!')