import timm
import torch
import warnings
import numpy as np
from core.model import *
from utils.utils import set_seeds, KNNGaussianBlur
from sklearn.metrics import roc_auc_score
from utils.au_pro_util import calculate_au_pro
from sklearn.decomposition import sparse_encode
from utils.visualize_util import visualize_smap_distribute, visualize_image_s_distribute
from ptflops import get_model_complexity_info

warnings.filterwarnings("ignore", category=RuntimeWarning) 


class Features(torch.nn.Module):
    def __init__(self, image_size=224, pro_limit=[0.3]):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rgb_feature_extractor = RGB_Model(device=self.device)
        self.rgb_feature_extractor.to(self.device)
        self.rgb_feature_extractor.freeze_parameters(layers=[], freeze_bn=True)

        self.image_size = image_size
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        set_seeds(0)
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.pro_limit = pro_limit

        self.weight = 0
        self.bias = 0
        self.origin_f_map = []
        self.rgb_f_idx_patch_lib = []
        self.sdf_patch_lib = []
        self.rgb_patch_lib = []
        self.initialize_score()

    def __call__(self, x):
        # Extract the desired feature maps using the backbone model.
        with torch.no_grad():
            feature_maps = self.rgb_feature_extractor(x)

        # feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps

    def initialize_score(self):
        # Image-Level
        self.image_preds = list()
        self.sdf_image_preds = list()
        self.rgb_image_preds = list()
        self.image_labels = list()

        # Pixel-Level
        self.pixel_preds = list()
        self.sdf_pixel_preds = list()
        self.rgb_pixel_preds = list()
        self.new_rgb_pixel_preds = list()
        self.pixel_labels = list()
        self.au_pro = dict()

    def get_result(self):
        predictions = np.asarray(self.pixel_preds).reshape(-1, self.image_size, self.image_size)
        return self.image_preds, predictions

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def count_your_model(model, input):
        flops, params = get_model_complexity_info(model, input, as_strings=True, print_per_layer_stat=True)
        print('Flops: ' + flops)
        print('Params: ' + params)

    def foreground_subsampling(self): 
        self.sdf_patch_lib = torch.cat(self.sdf_patch_lib, 0)
        self.rgb_patch_lib = torch.cat(self.rgb_patch_lib, 0)

        # Remove unused RGB features #
        self.origin_f_map = np.full(shape=(self.rgb_patch_lib.shape[0]),fill_value=-1)
        use_f_idices = torch.unique(torch.cat(self.rgb_f_idx_patch_lib,dim=0))
        self.rgb_patch_lib = self.rgb_patch_lib[use_f_idices]
        self.origin_f_map[use_f_idices] = np.array(range(use_f_idices.shape[0]),dtype=int)
        self.origin_f_map = torch.Tensor(self.origin_f_map).long()
        
    def Dict_compute_rgb_map(self, rgb_patch_28, rgb_features_indices, lib_idices, mode='testing'):
        rgb_lib = self.rgb_patch_lib.to(self.device)
        lib_idices = torch.unique(lib_idices.flatten()).tolist()
        s_map_size28 = torch.zeros(28*28)
        pdist = torch.nn.PairwiseDistance(p=2, eps= 1e-12)

        ### Get shape guide RGB features which stored in the memory ###
        rgb_indices_28 = []
        for patch_idx in lib_idices:
            _f_idx_28 = self.origin_f_map[self.rgb_f_idx_patch_lib[patch_idx]]
            rgb_indices_28.append(_f_idx_28)
            
        rgb_indices_28 = torch.unique(torch.cat(rgb_indices_28,dim=0))
        shape_guide_rgb_features_28 = rgb_lib[rgb_indices_28]

        ### Get foreground RGB features of test data ###
        foreground_rgb_indices_28 = torch.unique(torch.cat(rgb_features_indices,dim=0))
        rgb_patch_28 = rgb_patch_28[foreground_rgb_indices_28]
        
        ############## Feature size 28 ##############
        Dict_features_size28 = []
        dist_28 = torch.cdist(rgb_patch_28, shape_guide_rgb_features_28.to(self.device))
        knn_val, knn_idx = torch.topk(dist_28, k=5+1, largest=False)

        if mode == 'alignment':
            knn_idx = knn_idx[:, 1:]
            min_val_28 = knn_val[:, 1]
        elif mode == 'testing':
            knn_idx = knn_idx[:,:-1]
            min_val_28 = knn_val[:, 0]
            
        if True: # Using Dictionart if TRUE, otherwise using NN
            rgb_patch_28 = rgb_patch_28.cpu()
            shape_guide_rgb_features_28 = shape_guide_rgb_features_28.cpu()
            for patch in range(knn_idx.shape[0]):
                knn_features =  shape_guide_rgb_features_28[knn_idx[patch]]
                code_matrix = sparse_encode(X=rgb_patch_28[patch].view(1,-1), dictionary=knn_features, algorithm='omp', n_nonzero_coefs=3, alpha=1e-10)
                code_matrix = torch.Tensor(code_matrix)
                sparse_features = torch.matmul(code_matrix, knn_features) # Sparse representation test rgb feature using the training rgb features stored in the memory.
                Dict_features_size28.append(sparse_features)

            Dict_features_size28 = torch.cat(Dict_features_size28, 0)
            s_map_size28[foreground_rgb_indices_28] = pdist(rgb_patch_28, Dict_features_size28)
        else:
            s_map_size28[foreground_rgb_indices_28] = min_val_28.to('cpu')
            
        ############## Reshape anomaly score map ##############
        if mode == 'alignment':
            s = torch.max(s_map_size28)
        elif mode == 'testing':
            s = torch.max(s_map_size28)

        s_map_size28 = s_map_size28.view(1, 1, 28, 28)
        s_map_size28 = torch.nn.functional.interpolate(s_map_size28, size=(self.image_size, self.image_size), mode='bilinear', align_corners = False)
        s_map = s_map_size28.to('cpu')
       
        return s_map, s

    def Find_KNN_feature(self, feature, mode='testing'):
        Dict_features = []
        patch_lib = self.sdf_patch_lib.to(self.device)
        dist = torch.cdist(feature, patch_lib)
        _, knn_idx = torch.topk(dist, k=10+1, largest=False)

        knn_idx = knn_idx.cpu()
        if mode == 'alignment':
            knn_idx = knn_idx[:,1:]
        elif mode == 'testing':
            knn_idx = knn_idx[:,:-1]

        feature = feature.to('cpu')
        for patch in range(knn_idx.shape[0]):
            Knn_features = self.sdf_patch_lib[knn_idx[patch]]
            code_matrix = sparse_encode(X=feature[patch].view(1,-1), dictionary=Knn_features, algorithm='omp', n_nonzero_coefs=3, alpha=1e-10)
            code_matrix = torch.Tensor(code_matrix)
            sparse_feature = torch.matmul(code_matrix, Knn_features) # Sparse representation test rgb feature using the training rgb features stored in the memory.
            Dict_features.append(sparse_feature)
            
        Dict_features = torch.cat(Dict_features, 0)
        NN_feature = self.sdf_patch_lib[knn_idx[:, 0]]   # find the nearest neighbor feature
        pdist = torch.nn.PairwiseDistance(p=2, eps=1e-12)
        min_val = pdist(feature, Dict_features)
        s = torch.max(min_val) # Compute image level anomaly score #
        return NN_feature, Dict_features, knn_idx, s

    def cal_alignment(self, output_dir):
        # SDF distribution
        sdf_map = np.array(self.sdf_pixel_preds)
        non_zero_indice = np.nonzero(sdf_map)
        non_zero_sdf_map = sdf_map[non_zero_indice]
        sdf_mean = np.mean(non_zero_sdf_map)
        sdf_std = np.std(non_zero_sdf_map)
        sdf_lower = sdf_mean - 3 * sdf_std
        sdf_upper = sdf_mean + 3 * sdf_std
        # RGB distribution
        rgb_map = np.array(self.rgb_pixel_preds)
        non_zero_indice = np.nonzero(rgb_map)
        non_zero_rgb_map = rgb_map[non_zero_indice]
        rgb_mean = np.mean(non_zero_rgb_map)
        rgb_std = np.std(non_zero_rgb_map)
        rgb_lower = rgb_mean - 3 * rgb_std
        rgb_upper = rgb_mean + 3 * rgb_std
        
        self.weight = (sdf_upper - sdf_lower) / (rgb_upper - rgb_lower)
        self.bias = sdf_lower - self.weight * rgb_lower
        new_rgb_map = rgb_map * self.weight  + self.bias
        total_score = np.maximum(new_rgb_map, sdf_map)

        visualize_smap_distribute(total_score, sdf_map, rgb_map, new_rgb_map, self.image_size, output_dir)
        return self.weight, self.bias

    def cal_total_score(self, output_dir, method='RGB_SDF'):

        if method == 'RGB':
            image_preds = np.stack(self.rgb_image_preds)
            pixel_preds = np.array(self.rgb_pixel_preds)
        elif method == 'SDF':
            image_preds = np.stack(self.sdf_image_preds)
            pixel_preds = np.array(self.sdf_pixel_preds)
        else:
            image_preds = np.stack(self.image_preds)
            pixel_preds = np.array(self.pixel_preds)

        gts = np.array(self.pixel_labels).reshape(-1, self.image_size, self.image_size)
        predictions = pixel_preds.reshape(-1, self.image_size, self.image_size)

        # visualize the distribution of image score and pixel score
        if len(self.rgb_pixel_preds) != 0 and method == 'RGB_SDF':
            sdf_map = np.array(self.sdf_pixel_preds)
            rgb_map = np.array(self.rgb_pixel_preds)
            new_rgb_map = np.array(self.new_rgb_pixel_preds)
            sdf_s = np.array(self.sdf_image_preds)
            rgb_s = np.array(self.rgb_image_preds)
            label = np.array(self.image_labels)
            visualize_image_s_distribute(sdf_s, rgb_s, label, output_dir)
            visualize_smap_distribute(pixel_preds, sdf_map, rgb_map, new_rgb_map, self.image_size, output_dir)
       
        self.image_rocauc = roc_auc_score(self.image_labels, image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, pixel_preds)
        for pro_integration_limit in self.pro_limit:
            au_pro, _ = calculate_au_pro(gts, predictions, integration_limit=pro_integration_limit)
            self.au_pro[str(pro_integration_limit)] = au_pro

class RGB_Model(torch.nn.Module):
    def __init__(self, device, backbone_name='wide_resnet50_2', out_indices=(1, 2), checkpoint_path='',
                 pool_last=False):
        super().__init__()
        # Determine if to output features.
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, checkpoint_path=checkpoint_path,
                                          **kwargs)
        self.device = device
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None
    
    def forward(self, x):
        x = x.to(self.device)

        # Backbone forward pass.
        features = self.backbone(x)

        # Adaptive average pool over the last layer.
        if self.avg_pool:
            fmap = features[-1]
            fmap = self.avg_pool(fmap)
            fmap = torch.flatten(fmap, 1)
            features.append(fmap)
        
        return features

    def freeze_parameters(self, layers, freeze_bn=False):
        """ Freeze resent parameters. The layers which are not indicated in the layers list are freeze. """

        layers = [str(layer) for layer in layers]
        # Freeze first block.
        if '1' not in layers:
            if hasattr(self.backbone, 'conv1'):
                for p in self.backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'bn1'):
                for p in self.backbone.bn1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'layer1'):
                for p in self.backbone.layer1.parameters():
                    p.requires_grad = False

        # Freeze second block.
        if '2' not in layers:
            if hasattr(self.backbone, 'layer2'):
                for p in self.backbone.layer2.parameters():
                    p.requires_grad = False

        # Freeze third block.
        if '3' not in layers:
            if hasattr(self.backbone, 'layer3'):
                for p in self.backbone.layer3.parameters():
                    p.requires_grad = False

        # Freeze fourth block.
        if '4' not in layers:
            if hasattr(self.backbone, 'layer4'):
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = False

        # Freeze last FC layer.
        if '-1' not in layers:
            if hasattr(self.backbone, 'fc'):
                for p in self.backbone.fc.parameters():
                    p.requires_grad = False

        if freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb_model = RGB_Model(device)
    flops, params = get_model_complexity_info(rgb_model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('Flops: ' + flops)
    print('Params: ' + params)