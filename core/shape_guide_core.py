import numpy as np
import torch
from tqdm import tqdm
from core.rgb_sdf_feature import RGBSDFFeatures, SDFFeature
from core.data import *
from utils.utils import *

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
        rgb_method='Dict',
        k_number=1,
        dict_n_component=3,
        method_name=None
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
        self.method_name = method_name
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
        self.class_name = class_name
    
        self.datasets_path = conf.datasets_path
        self.grid_path = conf.grid_path

        self.parent_dir = conf.output_dir
        output_dir = os.path.join(conf.output_dir, class_name)
        self.pro_limit = pro_limit

        self.method_name = conf.method_name
        self.methods = RGBSDFFeatures(conf, pro_limit, output_dir)
        # Create class dir and pc dir
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        # create writer
        buf_size = 1  # Make 'testing_state' file to flush each output line regarding training.
        self.log_file = open(osp.join(output_dir, "testing_state.txt"), "a", buf_size)

    def fit(self):
        data_loader = get_data_loader("train", class_name=self.class_name, img_size=self.image_size, 
        datasets_path=self.datasets_path, grid_path=self.grid_path, shuffle=True)
        with torch.no_grad():
            for train_data_id, (sample, _) in enumerate(tqdm(data_loader, desc=f'Extracting train features for class {self.class_name}')):
                self.methods.add_sample_to_mem_bank(sample, train_data_id)

        print(f'\n\nRunning ForeGround Subsampling on class {self.class_name}...')
        self.methods.foreground_subsampling()

    def align(self):
        data_loader = get_data_loader("train", class_name=self.class_name, img_size=self.image_size, datasets_path=self.datasets_path, grid_path=self.grid_path, shuffle=True)
        with torch.no_grad():
            for align_data_id, (sample, _) in enumerate(tqdm(data_loader, desc=f'Extracting aligned features for class {self.class_name}')):
                if align_data_id < 25:
                    self.methods.predict_align_data(sample, align_data_id)
                else: 
                    break

        weight, bias = self.methods.cal_alignment()
        buf_size = 1
        log_file = open(osp.join(self.parent_dir, "alignment.txt"), "a", buf_size)
        log_file.write("'%s'\t:[%.16f, %.16f],\n" % (self.class_name, weight, bias))
        log_file.close()
    
    def evaluate(self):
        self.methods.initialize_score()
            
        test_loader = get_data_loader("test", class_name=self.class_name, img_size=self.image_size, 
        datasets_path=self.datasets_path, grid_path=self.grid_path)

        with torch.no_grad():
            for test_data_id, (sample, mask, label) in enumerate(tqdm(test_loader, desc=f'Extracting test features for class {self.class_name}')):
                self.methods.predict(sample, mask, label, test_data_id)

        # Just visualize RGB+SDF method and compute its threshold
        det_threshold, seg_threshold = self.methods.visualize_result()
        self.log_file.write('Optimal DET Threshold: {:.2f}\n'.format(det_threshold))
        self.log_file.write('Optimal SEG Threshold: {:.2f}\n'.format(seg_threshold))

        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        # Calculate Score of RGB, SDF, RGB+SDF
        for method_name in self.method_name:
            self.methods.cal_total_score(method=method_name)
            
            image_rocaucs[method_name] = round(self.methods.image_rocauc, 3)
            pixel_rocaucs[method_name] = round(self.methods.pixel_rocauc, 3)
            for integration_limit in self.pro_limit:
                au_pros[method_name + '_' + str(integration_limit)] = round(self.methods.au_pro[str(integration_limit)], 3)
            
            self.log_file.write(
                f'Class: {self.class_name}, {method_name} Image ROCAUC: {self.methods.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {self.methods.pixel_rocauc:.3f}, {method_name} \
                AU-PRO 0.3: {self.methods.au_pro[str(0.3)]:.3f}\n \
                AU-PRO 0.2: {self.methods.au_pro[str(0.2)]:.3f}\n \
                AU-PRO 0.1: {self.methods.au_pro[str(0.1)]:.3f}\n \
                AU-PRO 0.07: {self.methods.au_pro[str(0.07)]:.3f}\n \
                AU-PRO 0.05: {self.methods.au_pro[str(0.05)]:.3f}\n \
                AU-PRO 0.03: {self.methods.au_pro[str(0.03)]:.3f}\n \
                AU-PRO 0.01: {self.methods.au_pro[str(0.01)]:.3f}\n')

        self.log_file.close()
        return image_rocaucs, pixel_rocaucs, au_pros
