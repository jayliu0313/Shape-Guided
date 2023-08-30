import os
import glob
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils.mvtec3d_util import *
from utils.utils import * 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MVTec3D(Dataset):
    def __init__(self, split, class_name, img_size, datasets_path, grid_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(datasets_path, self.cls, split)
        self.npz_path = os.path.join(grid_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((self.size, self.size),transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])
   
class MVTec3DTrain(MVTec3D):
    def __init__(self, class_name, img_size, datasets_path, grid_path):
        super().__init__(split="train", class_name=class_name, img_size=img_size, datasets_path=datasets_path, grid_path=grid_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        npz_paths = glob.glob(os.path.join(self.npz_path, 'good', 'npz') + "/*.npz")
        rgb_paths.sort()
        tiff_paths.sort()
        npz_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths, npz_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        npz_path = img_path[2]
        #load image data
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)
        #load npz data
        points_gt_all, points_idx_all , points_tran_all = data_process(npz_path)
        return (img, points_gt_all, points_idx_all), label

class MVTec3DTest(MVTec3D):
    def __init__(self, class_name, img_size, datasets_path, grid_path):
        super().__init__(split="test", class_name=class_name, img_size=img_size, datasets_path=datasets_path, grid_path=grid_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                npz_dir_paths = glob.glob(os.path.join(self.npz_path, defect_type, 'npz') + "/*.npz")
                rgb_paths.sort()
                tiff_paths.sort()
                npz_dir_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths, npz_dir_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                npz_dir_paths = glob.glob(os.path.join(self.npz_path, defect_type, 'npz') + "/*.npz")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                npz_dir_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths, npz_dir_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))
                
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        npz_path = img_path[2]

        #load image data
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        #load points cloud data
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        # resized_organized_pc = resize_organized_pc(organized_pc)
        #load npz data
        points_gt_all, points_idx_all , points_tran_all = data_process(npz_path)
        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)
        return (img, points_gt_all, points_idx_all), gt[:1], label

class MVTec3DValidation(MVTec3D):
    def __init__(self, class_name, img_size, datasets_path, grid_path):
        super().__init__(split="validation", class_name=class_name, img_size=img_size, datasets_path=datasets_path, grid_path=grid_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                npz_dir_paths = glob.glob(os.path.join(self.npz_path, defect_type, 'npz') + "/*.npz")
                rgb_paths.sort()
                tiff_paths.sort()
                npz_dir_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths, npz_dir_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                npz_dir_paths = glob.glob(os.path.join(self.npz_path, defect_type, 'npz') + "/*.npz")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                npz_dir_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths, npz_dir_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))
                
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        npz_path = img_path[2]

        #load image data
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)
        
        #load points cloud data
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)
        #load npz data
        points_gt_all, points_idx_all , points_tran_all= data_process(npz_path)
        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return (img, resized_organized_pc, resized_depth_map_3channel, points_gt_all, points_idx_all, points_tran_all), gt[:1], label

class MVTec3DPreTrain(Dataset):
    def __init__(self, class_name, point_num, sample_size, grid_path):
        self.class_name = class_name
        self.point_num = point_num
        self.sample_size = sample_size
        self.grid_path = grid_path
        self.points_all, self.samples_all= self.load_dataset()  # self.labels => good : 0, anomaly : 1
        print('# # # # # # # Total Number of Patches:', len(self.points_all), '# # # # # # #')

    def load_dataset(self):
        npz_paths = glob.glob(os.path.join(self.grid_path, 'PRETRAIN_DATA', self.class_name) + "/*.npz")
        npz_paths.sort()
        samples_all = []
        points_all = []

        for npz_path in tqdm(npz_paths, desc='Load Data for Pre-Training'):
            load_data = np.load(npz_path, allow_pickle=True)
            samples_set = np.asarray(load_data['samples_all'])    # all of the sample points
            points_set = np.asarray(load_data['points_all'])      # the gt of sample points

            #print('Load npz path:', npz_path)
            #print('Number of patches:', points_set.shape[0])

            for patch in range(points_set.shape[0]):
                point, sample = points_set[patch] , samples_set[patch]
                points_all.append(point)
                samples_all.append(sample)
        
        return points_all, samples_all

    def __len__(self):
        return len(self.points_all)

    def __getitem__(self, idx):
        points, samples = self.points_all[idx], self.samples_all[idx]
        rt = random.randint(0,self.sample_size - 1)
        points = points.reshape(self.sample_size, self.point_num, 3)[rt,:,:]
        samples = samples.reshape(self.sample_size, self.point_num, 3)[rt,:,:]
        return points, samples

class MVTec3DTestRGB(Dataset):
    def __init__(self, img_size, datasets_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size),transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])
        # self.img_size = img_size
        self.datasets_path = datasets_path
        self.img_paths, self.cls_labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        class_name = [
            "bagel",
            "cable_gland",
            "carrot",
            "cookie",
            "dowel",
            "foam",
            "peach",
            "potato",
            "rope",
            "tire"
        ]
        img_tot_paths = []
        img_cls_label = []
        for i in range(len(class_name)):
            img_path = os.path.join(self.datasets_path, class_name[i], 'test')
            defect_types = os.listdir(img_path)
            for defect_type in defect_types:
                rgb_paths = glob.glob(os.path.join(img_path, defect_type, 'rgb') + "/*.png")
                sample_paths = list(zip(rgb_paths))
                img_tot_paths.extend(sample_paths)
                img_cls_label.extend([i] * len(sample_paths))
        return img_tot_paths, img_cls_label

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path = self.img_paths[idx][0]
        cls_label = self.cls_labels[idx]
        
        #load image data
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)
        return img, torch.eye(10)[cls_label]
 
class MVTec3DTrainRGB(Dataset):
    def __init__(self, img_size, datasets_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size),transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])
        # self.img_size = img_size
        self.datasets_path = datasets_path
        self.img_paths, self.cls_labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        class_name = [
            "bagel",
            "cable_gland",
            "carrot",
            "cookie",
            "dowel",
            "foam",
            "peach",
            "potato",
            "rope",
            "tire"
        ]
        img_tot_paths = []
        img_cls_label = []
        for i in range(len(class_name)):
            img_path = os.path.join(self.datasets_path, class_name[i], 'train')
            rgb_paths = glob.glob(os.path.join(img_path, 'good', 'rgb') + "/*.png")
            sample_paths = list(zip(rgb_paths))
            img_tot_paths.extend(sample_paths)
            img_cls_label.extend([i] * len(sample_paths))
        return img_tot_paths, img_cls_label

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path = self.img_paths[idx][0]
        cls_label = self.cls_labels[idx]
        
        #load image data
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)
        return img, torch.eye(10)[cls_label]

def data_process(npz_dir_path):
    points_gt_all = []
    points_idx_all = []
    points_tran_all = []
    files_path = npz_dir_path
    if(os.path.exists(files_path)):
        load_data = np.load(files_path)
        points_gt_set = np.asarray(load_data['points_gt'])
        points_idx_set = np.asarray(load_data['points_idx'])
        
        #print('data patches number:',points_gt_set.shape[0])
        for patch in range(points_gt_set.shape[0]):
        
            points_gt = points_gt_set[patch]
            points_idx = points_idx_set[patch]
            points_gt, points_tran = normal_points(points_gt, True)

            points_gt_all.append(points_gt)
            points_idx_all.append(points_idx)
            points_tran_all.append(points_tran)

    return points_gt_all, points_idx_all, points_tran_all

def get_pretrain_data_loader(cls, conf):
    dataset = MVTec3DPreTrain(class_name=cls, point_num=conf.POINT_NUM, sample_size=conf.sampled_size, grid_path=conf.grid_path)
    data_loader = DataLoader(dataset=dataset, batch_size=conf.BS, shuffle=True, num_workers=1, drop_last=False, pin_memory=True)
    return data_loader

def get_data_loader(split, class_name, img_size, datasets_path, grid_path, shuffle=False):
    if split in ['train']:
        dataset = MVTec3DTrain(class_name=class_name, img_size=img_size, datasets_path=datasets_path, grid_path=grid_path)
    elif split in ['test']:
        dataset = MVTec3DTest(class_name=class_name, img_size=img_size, datasets_path=datasets_path, grid_path=grid_path)
    elif split in ['validation']:
        dataset = MVTec3DValidation(class_name=class_name, img_size=img_size, datasets_path=datasets_path, grid_path=grid_path)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=1, drop_last=False, pin_memory=True)
    return data_loader

def get_rgb_data(split, img_size, datasets_path):
    if split == 'train':
        dataset = MVTec3DTrainRGB(img_size=img_size, datasets_path=datasets_path)
        #data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False, pin_memory=True)
    elif split == 'test':
        dataset = MVTec3DTestRGB(img_size=img_size, datasets_path=datasets_path)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False, pin_memory=True)
    return dataset, data_loader

def pretrain_normal_points(ps_gt, ps, translation=False):
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

    if(translation):
        t = np.mean(ps_gt,axis = 0)
        ps_gt = ps_gt - t
        ps = ps - t
    
    return ps_gt, ps, (t, tt)

def normal_points(ps_gt, translation=False): 
    tt =  0
    if((np.max(ps_gt[:,0])-np.min(ps_gt[:,0]))>(np.max(ps_gt[:,1])-np.min(ps_gt[:,1]))):
        tt = (np.max(ps_gt[:,0])-np.min(ps_gt[:,0]))
    else:
        tt = (np.max(ps_gt[:,1])-np.min(ps_gt[:,1]))
    if(tt < (np.max(ps_gt[:,2])-np.min(ps_gt[:,2]))):
        tt = (np.max(ps_gt[:,2])-np.min(ps_gt[:,2]))
     
    tt = 10/(10*tt)
    ps_gt = ps_gt*tt

    if(translation):
        t = np.mean(ps_gt,axis = 0)
        ps_gt = ps_gt - t

    return ps_gt, (t , tt)
