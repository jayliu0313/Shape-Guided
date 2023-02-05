import timm
import numpy as np
from timm.optim import optim_factory
import torch
import torch.nn as nn 
import os
from core.data import get_rgb_data
from types import SimpleNamespace
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
DEBUG = False
lr = 0.001

parser.add_argument('--image_size', type=int, default=224, help="Reduced 800*800 pc to n*n by interpolate")
# something about path 
if DEBUG:
    parser.add_argument('--datasets_path', type=str, default="/mnt/home_6T/public/samchu0218/Test/not_cut/", help="The dir path of mvtec3D-AD dataset")
    parser.add_argument('--grid_path', type=str, default="/mnt/home_6T/public/samchu0218/Test/cut/imgsize224_knn500/gm10_sampled20/", help="The dir path of grid you cut, it would include training npz, testing npz")
else:
    parser.add_argument('--datasets_path', type=str, default="/mnt/home_6T/public/samchu0218/Datasets/mvtec3d_preprocessing/")                   #"The dir path of mvtec3D-AD dataset"
    parser.add_argument('--grid_path', type=str, default="/mnt/home_6T/public/samchu0218/Datasets/mvtec3d_cut_grid/imgsize224_knn500/gm10_sampled20/")                 #The dir path of grid you cut, it would include training npz, testing npz
parser.add_argument('--CUDA', type=int, default=1, help="choose the device of CUDA")
a = parser.parse_args()
cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx

if DEBUG:
    class_name = ["cable_gland"]
else:
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
        ]  # load category

image_size = a.image_size
datasets_path = a.datasets_path
grid_path = a.grid_path

class Model(torch.nn.Module):
    def __init__(self, device, backbone_name='wide_resnet50_2', out_indices=False, checkpoint_path='',
                 pool_last=False):
        super().__init__()
        # Determine if to output features.
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, checkpoint_path=checkpoint_path, num_classes=10)
        self.device = device
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None
        print(self.device)

    def forward(self, x):
        x = x.to(self.device)
        # print(x)
        # Backbone forward pass.
        output = self.backbone(x)
        
        # Adaptive average pool over the last layer.
        # if self.avg_pool:
        #     fmap = features[-1]
        #     fmap = self.avg_pool(fmap)
        #     fmap = torch.flatten(fmap, 1)
        #     features.append(fmap)

        return output

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

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def training_one_step(dataset, args, loader, model, loss_fn = nn.CrossEntropyLoss(), **optim_kwargs):
    accuracy = 0
    optimizer = optim_factory.create_optimizer(args, model, **optim_kwargs)
    loss_avg = AverageMeter()
    # losses = []
    tk0 = tqdm(enumerate(loader), total=len(loader))
    for i, (sample, targets) in tk0:
        targets = targets.cuda()
        preds = model(sample)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_avg.update(loss.item(), loader.batch_size)
        output = torch.argmax(preds)
        target = torch.argmax(targets)
        accuracy += (output == target).float().sum()
        # print(preds)
        # output = torch.argmax(preds)
        # print(output)
        # losses.append(loss_avg.avg)
        tk0.set_postfix(loss=loss.item())
    accuracy = 100 * accuracy / len(dataset)
    return loss_avg.avg, accuracy

def validation(dataset, loader, model, loss_fn = nn.CrossEntropyLoss()):
    accuracy = 0
    model.eval()
    loss_avg = AverageMeter()
    # losses = []
    tk0 = tqdm(enumerate(loader), total=len(loader))
    for i, (sample, targets) in tk0:
        targets = targets.cuda()
        preds = model(sample)
        loss = loss_fn(preds, targets)
        loss_avg.update(loss.item(), loader.batch_size)
        output = torch.argmax(preds)
        target = torch.argmax(targets)
        # print("targets", targets)
        # print("target", target)
        # print("output", output)
        # print("pred", preds)
        accuracy += (output == target).float().sum()
        tk0.set_postfix(loss=loss.item())
    accuracy = 100 * accuracy / len(dataset)
    # print(f'Validation Loss: {loss_avg.avg:.4f}, Accuracy: {accuracy:.4f}')
    return loss_avg.avg, accuracy


args = SimpleNamespace()
args.weight_decay = 0 
args.lr = lr
args.momentum = 0.9
args.opt = 'Adam'

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model(device=device, pool_last=True)
model.to(device)
model.freeze_parameters(layers=[-1], freeze_bn=True)
train_set, train_loader = get_rgb_data("train", image_size, datasets_path)
test_set, test_loader = get_rgb_data("test", image_size, datasets_path)
epochs = 10

for epoch in range(epochs):
    avg_loss, accuracy = training_one_step(dataset=train_set, args=args, loader=train_loader, model=model)
    print(f'[{epoch + 1}] training average loss:{avg_loss}, Accuracy:{accuracy}')
    # if epoch % 5 == 0 and epoch != 0:
    avg_loss, accuracy = validation(dataset=test_set, loader=test_loader, model=model)
    print(f'[{epoch + 1}] validation average loss:{avg_loss}, Accuracy:{accuracy}')
