import numpy as np
import torch
from tqdm import tqdm
from core.rgb_sdf_feature import RGBSDFFeatures
from core.data import *
from utils.utils import *
from utils.visualize_util import visualization

class Configuration(object):
    def __init__(
        self,
        image_size=224,
        sampled_size=20,
        POINT_NUM=500,
        group_mul=10,
        BS=1,
        datasets_path=None,
        grid_path=None,
        ckpt_dir=None,
        output_dir=None,
        LR=0.0001,
        classes=None,
        sdf=None,
        rgb_method='Dict',
        k_number=1,
        dict_n_component=3,
        method=None
    ):
        self.image_size = image_size
        self.sampled_size = sampled_size
        self.POINT_NUM = POINT_NUM
        self.group_mul = group_mul
        self.BS = BS
        self.datasets_path = datasets_path
        self.grid_path = grid_path
        self.ckpt_dir = ckpt_dir
        self.output_dir = output_dir
        self.LR = LR
        self.classes = classes
        self.sdf = sdf
        self.method = method
        self.rgb_method = rgb_method
        self.k_number = k_number
        self.dict_n_component = dict_n_component
    
    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    def __str__(self):
        keys = list(self.__dict__.keys())
        vals = list(self.__dict__.values())
        index = np.argsort(keys)
        res = ""
        for i in index:
            if callable(vals[i]):
                v = vals[i].__name__
            else:
                v = str(vals[i])
            res += "%30s: %s\n" % (str(keys[i]), v)
        return res

    def save(self, file_name):
        pickle_data(file_name + ".pickle", self)
        with open(file_name + ".txt", "w") as fout:
            fout.write(self.__str__())
            
    @staticmethod
    def load(file_name):
        return next(unpickle_data(file_name + ".pickle"))

class ShapeGuide():
    def __init__(self, conf, class_name, pro_limit):
        self.image_size = conf.image_size
        self.methods = RGBSDFFeatures(image_size=conf.image_size, pro_limit=pro_limit)
        self.class_name = class_name
        self.POINT_NUM = conf.POINT_NUM
        self.BS = conf.BS
        self.datasets_path = conf.datasets_path
        self.grid_path = conf.grid_path
        self.ckpt_dir = conf.ckpt_dir
        self.parent_dir = conf.output_dir
        self.output_dir = os.path.join(conf.output_dir, class_name)
        self.sdf = conf.sdf
        self.pro_limit = pro_limit
        
        # Create class dir and pc dir
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # create writer
        buf_size = 1  # Make 'testing_state' file to flush each output line regarding training.
        self.sdf.log_file = open(osp.join(self.output_dir, "testing_state.txt"), "a", buf_size)

    def fit(self):
        self.data_loader = get_data_loader("train", class_name=self.class_name, img_size=self.image_size, 
        datasets_path=self.datasets_path, grid_path=self.grid_path, shuffle=True)
        with torch.no_grad():
            for train_data_id, (sample, _) in enumerate(tqdm(self.data_loader, desc=f'Extracting train features for class {self.class_name}')):
                self.methods.add_sample_to_mem_bank(self.sdf, sample, train_data_id)

        print(f'\n\nRunning ForeGround Subsampling on class {self.class_name}...')
        self.methods.foreground_subsampling()

    def align(self):
        with torch.no_grad():
            for align_data_id, (sample, _) in enumerate(tqdm(self.data_loader, desc=f'Extracting aligned features for class {self.class_name}')):
                if align_data_id < 25:
                    self.methods.predict_align_data(self.sdf, sample, None, align_data_id)
                else: 
                    break

        weight, bias = self.methods.cal_alignment(self.output_dir)
        buf_size = 1
        log_file = open(osp.join(self.parent_dir, "alignment.txt"), "a", buf_size)
        log_file.write("'%s'\t:[%.16f, %.16f],\n" % (self.class_name, weight, bias))
        log_file.close()
    
    def evaluate(self):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()

        image_list = list()
        gt_label_list = list()
        gt_mask_list = list()
        self.methods.initialize_score()
            
        test_loader = get_data_loader("test", class_name=self.class_name, img_size=self.image_size, 
        datasets_path=self.datasets_path, grid_path=self.grid_path)

        with torch.no_grad():
            for test_data_id, (sample, mask, label) in enumerate(tqdm(test_loader, desc=f'Extracting test features for class {self.class_name}')):
                self.methods.predict(self.sdf, sample, mask, label, test_data_id)
                image_list.extend(t2np(sample[0]))
                gt_label_list.extend(t2np(label))
                gt_mask_list.extend(t2np(mask))
      
        gt_label_list = np.asarray(gt_label_list, dtype=np.bool)
        gt_mask_list = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
        image_score_list, score_map_list = self.methods.get_result()
        visualization(image_list, gt_label_list, image_score_list, gt_mask_list, score_map_list, self.output_dir, self.sdf.log_file)
        
        for method_name in self.methods.method:
            self.methods.cal_total_score(self.output_dir, method=method_name)
            image_rocaucs[method_name] = round(self.methods.image_rocauc, 3)
            pixel_rocaucs[method_name] = round(self.methods.pixel_rocauc, 3)
            for integration_limit in self.pro_limit:
                au_pros[method_name + '_' + str(integration_limit)] = round(self.methods.au_pro[str(integration_limit)], 3)
            
            self.sdf.log_file.write(
                f'Class: {self.class_name}, {method_name} Image ROCAUC: {self.methods.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {self.methods.pixel_rocauc:.3f}, {method_name} \
                AU-PRO 0.3: {self.methods.au_pro[str(0.3)]:.3f}\n \
                AU-PRO 0.2: {self.methods.au_pro[str(0.2)]:.3f}\n \
                AU-PRO 0.1: {self.methods.au_pro[str(0.1)]:.3f}\n \
                AU-PRO 0.07: {self.methods.au_pro[str(0.07)]:.3f}\n \
                AU-PRO 0.05: {self.methods.au_pro[str(0.05)]:.3f}\n \
                AU-PRO 0.03: {self.methods.au_pro[str(0.03)]:.3f}\n \
                AU-PRO 0.01: {self.methods.au_pro[str(0.01)]:.3f}\n')

        self.sdf.log_file.close()
        return image_rocaucs, pixel_rocaucs, au_pros
