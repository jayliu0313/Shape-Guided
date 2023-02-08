import torch
import numpy as np
from core.features import Features
from core.model import *
from utils.utils import *

RESULT_DIR = 'Save_PC_Result'

class RGBSDF(object):
    def __init__(self, image_size, BS, POINT_NUM, ckpt_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("SDF Device:", self.device)
        self.image_size = image_size
        self.BS = BS
        self.POINT_NUM = POINT_NUM

        # Launch the session
        self.sdf_model = SDF_Model(self.POINT_NUM).to(self.device)
        self.sdf_model.eval()
        
        # load check point
        checkpoint = torch.load(ckpt_dir, map_location=self.device)
        self.sdf_model.load_state_dict(checkpoint['sdf_model'])

    def get_feature(self, points_all, points_idx, data_id, split='test'):
        
        total_feature = None
        total_rgb_feature_indices = []
        
        for patch in range(len(points_all)):

            points = points_all[patch].reshape(-1, self.POINT_NUM, 3)
            indices = points_idx[patch].reshape(self.POINT_NUM)
            rgb_f_indices = get_relative_rgb_f_indices(indices, self.image_size, 28)
            feature = self.sdf_model.get_feature(points.to(self.device))
            
            if patch == 0:
                total_feature = feature
            else:
                total_feature = torch.cat((total_feature, feature), 0)
                
            if split == 'test':
                total_rgb_feature_indices.append(rgb_f_indices)
            elif split == 'train':
                total_rgb_feature_indices.append(data_id * 784 + rgb_f_indices)
            else:
                return KeyError
            
        return total_feature, total_rgb_feature_indices

    def get_score_map(self, feature, points_all, points_idx):

        s_map = np.full((self.image_size*self.image_size), 0, dtype=float)
        for patch in range(len(points_all)):

            if points_all[patch].shape[1] != self.POINT_NUM:
                print('Error!',points_all[patch].shape)
                continue
                    
            points = points_all[patch].reshape(-1, self.POINT_NUM, 3)
            indices = points_idx[patch].reshape(-1, self.POINT_NUM)

            point_feature = torch.tile(torch.unsqueeze(feature[patch], 0), [1, self.POINT_NUM, 1])
            index = indices[0].reshape(self.POINT_NUM)
            point_target = points[0,:].reshape(self.BS, self.POINT_NUM, 3)
            point_target = point_target.to(self.device)
            point_feature = point_feature.to(self.device)
            sdf_c = self.sdf_model.get_sdf(point_feature, point_target)

            sdf_c = np.abs(sdf_c.detach().cpu().numpy().reshape(-1))
            index = index.cpu().numpy()

            tmp_map = np.full((self.image_size*self.image_size), 0, dtype=float)
            for L in range(sdf_c.shape[0]):
                tmp_map[index[L]] = sdf_c[L]
                if(s_map[index[L]] == 0) or (s_map[index[L]] > sdf_c[L]):
                    s_map[index[L]] = sdf_c[L]

        s_map = s_map.reshape(1, 1, self.image_size, self.image_size)
        s_map = torch.tensor(s_map)
        return s_map

class RGBSDFFeatures(Features):
    def __init__(self, image_size=224, pro_limit=[0.3]):
        self.method = ['RGB_SDF', 'RGB', 'SDF']
        super().__init__(image_size, pro_limit)
        
    def add_sample_to_mem_bank(self, sdf, sample, train_data_id):
        ############### RGB PATCH ###############
        rgb_feature_maps = self(sample[0])
        
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        rgb_resized_maps = [self.resize(self.average(fmap)) for fmap in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_resized_maps, 1)
        rgb_patch_size28 = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        ############### END RGB PATCH ###############

        ############### PCP PATCH ###############
        sdf_feature, rgb_feature_indices_patch = sdf.get_feature(sample[1], sample[2], train_data_id, 'train')
        ############### END PCP PATCH ###############
        self.sdf_patch_lib.append(sdf_feature.to('cpu'))
        self.rgb_patch_lib.append(rgb_patch_size28.to('cpu'))
        self.rgb_f_idx_patch_lib.extend(rgb_feature_indices_patch)

    def predict(self, sdf, sample, mask, label, test_data_id):
        
        ############### PCP PATCH ###############
        feature, rgb_features_indices = sdf.get_feature(sample[1], sample[2], test_data_id, 'test')
        NN_feature, Dict_features, lib_idices, sdf_s = self.Find_KNN_feature(feature)
        sdf_map = sdf.get_score_map(Dict_features, sample[1], sample[2])
        ############### END PCP PATCH ###########

        ############### RGB PATCH ###############
        rgb_feature_maps = self(sample[0])
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        rgb_resized_maps = [self.resize(self.average(fmap)) for fmap in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_resized_maps, 1)
        rgb_features_size28 = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_map, rgb_s = self.Dict_compute_rgb_map(rgb_features_size28, rgb_features_indices, lib_idices, mode='testing')
        ############### END RGB PATCH ###########
        
        image_score = sdf_s * rgb_s
        new_rgb_map = rgb_map * self.weight + self.bias
        new_rgb_map = torch.clip(new_rgb_map, min=0, max=new_rgb_map.max())
        pixel_map = torch.maximum(new_rgb_map, sdf_map)

        sdf_map = self.blur(sdf_map)
        rgb_map = self.blur(rgb_map)
        new_rgb_map = self.blur(new_rgb_map)
        pixel_map = self.blur(pixel_map)

        ##### Record Image Level Score #####
        self.image_labels.append(label)
        self.sdf_image_preds.append(sdf_s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        self.image_preds.append((image_score).numpy())
        
        ##### Record Pixel Level Score #####
        self.pixel_labels.extend(mask.flatten().numpy())
        self.sdf_pixel_preds.extend(sdf_map.flatten().numpy())
        self.rgb_pixel_preds.extend(rgb_map.flatten().numpy())
        self.new_rgb_pixel_preds.extend(new_rgb_map.flatten().numpy())
        self.pixel_preds.extend(pixel_map.flatten().numpy())

    def predict_align_data(self, sdf, sample, label, test_data_id):
        ############### PCP PATCH ###############
        feature, rgb_features_indices = sdf.get_feature(sample[1], sample[2], test_data_id, 'test')
        NN_feature, Dict_features, lib_idices, sdf_s = self.Find_KNN_feature(feature, mode='alignment')
        sdf_map = sdf.get_score_map(Dict_features, sample[1], sample[2])
        ############### END PCP PATCH ###########

        ############### RGB PATCH ###############
        rgb_feature_maps = self(sample[0])

        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        rgb_resized_maps = [self.resize(self.average(fmap)) for fmap in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_resized_maps, 1)
        rgb_features_size28 = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_map, rgb_s = self.Dict_compute_rgb_map(rgb_features_size28, rgb_features_indices, lib_idices, mode='alignment')
        ############### END RGB PATCH ###########

        rgb_map = self.blur(rgb_map)
        sdf_map = self.blur(sdf_map)
        
        # image_level
        self.image_labels.append(label)
        self.sdf_image_preds.append(sdf_s.numpy())
        self.rgb_image_preds.append(rgb_s.numpy())
        # pixel_level
        self.rgb_pixel_preds.extend(rgb_map.flatten().numpy())
        self.sdf_pixel_preds.extend(sdf_map.flatten().numpy())