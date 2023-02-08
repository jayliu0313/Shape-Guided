import os
import math
import numpy as np
import os.path as osp
import random
import torch
import open3d as o3d
from torchvision import transforms
import torch.nn.functional as F
from PIL import ImageFilter
# configuration
from six.moves import cPickle
from scipy.spatial import cKDTree

# utils for configuration
def create_dir(dir_path):
    """ Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file.
    """
    myFile = open(file_name, "wb")
    cPickle.dump(len(args), myFile, protocol=2)
    for item in args:
        cPickle.dump(item, myFile, protocol=2)
    myFile.close()

def unpickle_data(file_name):
    """Restore data previously saved with pickle_data().
    """
    inFile = open(file_name, "rb")
    size = cPickle.load(inFile)
    for _ in range(size):
        yield cPickle.load(inFile)
    inFile.close()
    
np.random.seed(0)
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))

def save_pc(inputFile, save_path, output_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file = np.asarray(inputFile)
    file = file.reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(file)

    save_path = os.path.join(save_path,output_name)
    o3d.io.write_point_cloud(save_path + '.ply', pcd)

def output_txt(input_data, fname, fmt='%f'):
    """ Generate a txt file of input_data.
    """
    print('save txt of:', fname)
    new_array = np.array(input_data)
    np.savetxt('./' + fname + '.txt', new_array)

def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def denormal_points(points, translation):
    """Denormalize input points to recover the original shape through the translation information.
    """
    t = translation[0]
    tt = 10.0 / (10.0 * translation[1])
    pc = points.reshape(-1,3)
    pc = pc + t
    pc = pc * tt
    return pc

def list_join(A_list,B_list):
    new_list = []
    for a_item in A_list:
        for b_item in B_list:
            new_list.append(a_item + '_' + b_item)
    return new_list

# For Debug
def revise_NN(gt, sample):

    gt = np.array(gt).reshape(-1,3)
    sample = np.array(sample).reshape(-1,3)
    gt_origin = gt

    gt = np.unique(gt,axis=0)
    gt_kd_tree = cKDTree(gt)
    _ , vertex_ids = gt_kd_tree.query(sample, p=2, k = 1)
    vertex_ids = np.asarray(vertex_ids)
    result = gt[vertex_ids].reshape(-1,3)
    if(np.array_equal(gt_origin,result) == False):
        print("test_NN Error")
    return result

def random_sample_points(target_point_clouds, query_num):

    sample = []
    target_point_clouds = t2np(target_point_clouds).reshape(-1,3)
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
        
    sample = np.asarray(sample).reshape(-1,3)
    return sample

def pc_normalization(point_cloud):
    pointcloud_s = point_cloud.reshape(-1,3)
    pointcloud_s_t = pointcloud_s - (np.array([np.mean(pointcloud_s[:,0]),np.mean(pointcloud_s[:,1]),np.mean(pointcloud_s[:,2])]))
    pointcloud_s_t = pointcloud_s_t / (np.array([np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,1]) - np.min(pointcloud_s[:,1]), np.max(pointcloud_s[:,2]) - np.min(pointcloud_s[:,2])]))
    pointcloud_s = pointcloud_s_t
    return pointcloud_s

class KNNGaussianBlur(torch.nn.Module):
    def __init__(self, radius : int = 4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        return final_map

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def normalize(x):
    return (x - x.mean()) / x.std()

def show_bounding_box(point_cloud):
    pc = point_cloud.reshape(-1,3)
    #pc = t2np(point_cloud).reshape(-1,3)
    print('PC shape: ', pc.shape)
    print(f'X max: {np.max(pc[:,0])}, min: {np.min(pc[:,0])}')
    print(f'Y max: {np.max(pc[:,1])}, min: {np.min(pc[:,1])}')
    print(f'Z max: {np.max(pc[:,2])}, min: {np.min(pc[:,2])}')
    print('')

def get_relative_rgb_f_indices(target_pc_idices, img_size=224, f_size=28):
    scale = int(img_size / f_size)
    row = torch.div(target_pc_idices,img_size,rounding_mode='floor')
    col = target_pc_idices % img_size
    rgb_f_row = torch.div(row,scale,rounding_mode='floor')
    rgb_f_col = torch.div(col,scale,rounding_mode='floor')
    rgb_f_indices = rgb_f_row * f_size + rgb_f_col
    rgb_f_indices = torch.unique(rgb_f_indices)

    # More Background feature #
    B = 2
    rgb_f_indices = torch.cat([rgb_f_indices+B,rgb_f_indices-B,rgb_f_indices+28*B,rgb_f_indices-28*B],dim=0)
    rgb_f_indices[rgb_f_indices<0] = torch.max(rgb_f_indices)
    rgb_f_indices[rgb_f_indices>783] = torch.min(rgb_f_indices)
    rgb_f_indices = torch.unique(rgb_f_indices)

    return rgb_f_indices
    


