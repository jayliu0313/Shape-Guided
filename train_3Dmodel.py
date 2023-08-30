# -*- coding: utf-8 -*-
import os
import datetime
import os.path as osp

import argparse
from torch import optim
from utils.mvtec3d_util import *
from core.shape_guide_core import Configuration
from core.model import SDF_Model
from core.data import get_pretrain_data_loader
from tqdm import tqdm
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--point_num', type=int, default=500)  # For one local region include how many points, and it would input to the model
parser.add_argument('--group_mul', type=int, default=5)
parser.add_argument('--sampled_size', type=int, default=20)
parser.add_argument('--grid_path', type=str, default = "<grid_dir>", help="The dir path of grid you cut")
parser.add_argument('--ckpt_path', type=str, default = "./checkpoint") # auto recreate if exist
CKPT_FILENAME = "-knn500"
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--learning_rate", type=int, default=0.0001)


classes = [
"*"
]  # load category

a = parser.parse_args()
cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx

conf = Configuration(
    image_size = a.image_size,
    sampled_size = a.sampled_size,
    POINT_NUM = a.point_num,
    group_mul = a.group_mul,
    BS = a.batch_size,
    grid_path = a.grid_path,
    classes = classes,
    epoch = a.epoch,
    LR = a.learning_rate
)

# create the ckpt dir and conf file
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
conf.ckpt_path = os.path.join(a.ckpt_path, time) + CKPT_FILENAME
if not osp.exists(conf.ckpt_path):
    os.makedirs(conf.ckpt_path)
conf.save(os.path.join(conf.ckpt_path, "Congiguration"))

class Pretraining():
    def __init__(self, conf):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        # configuration
        self.BS = conf.BS
        self.POINT_NUM = conf.POINT_NUM
        self.ckpt_path = conf.ckpt_path
        learning_rate = conf.LR
        self.epoch = conf.epoch
        self.current_iter = 0

        # Initialize Model
        self.sdf_model = SDF_Model(self.POINT_NUM).to(self.device)
        self.optimizer = optim.Adam(self.sdf_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    def train(self, pretrain_loader):
        buf_size = 1  # Make 'training_stats' file to flush each output line regarding training.
        log_file = open(osp.join(self.ckpt_path, "train_states.txt"), "a", buf_size)
        for epoch in range(self.epoch + 1):
            loss_list = []
            for points, samples in tqdm(pretrain_loader, desc=f'Pre-Training Epoch {epoch}'):
                if points.shape[0] != self.BS:
                    continue
                point = points.reshape(self.BS, self.POINT_NUM, 3)
                point = point.to(torch.float32).cuda()
                sample = samples.reshape(self.BS, self.POINT_NUM, 3)
                sample = sample.to(torch.float32).cuda()

                g_points = self.sdf_model(point, sample)
                loss = torch.linalg.norm((point - g_points), ord=2, dim=-1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.detach().cpu().numpy())

            print('Loss:', sum(loss_list) / len(loss_list))
            if(epoch % 100 == 0):
                print('<Save Model> Epoch:', epoch ,' Loss:',loss.detach().cpu().numpy())
                self.save_checkpoint()
            if log_file is not None:
                log_file.write(
                    "epoch:%04d\tloss:%.9f\n" % (epoch, sum(loss_list)/len(loss_list))
                )
            self.current_iter += 1 
        log_file.close()
        self.save_checkpoint()

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.ckpt_path, checkpoint_name), map_location=self.device)
        print(os.path.join(self.ckpt_path, 'checkpoints', checkpoint_name))
        self.sdf_model.load_state_dict(checkpoint['sdf_model'])
        self.current_iter = checkpoint['current_iteration']
        
    def save_checkpoint(self):
        checkpoint = {
            'sdf_model': self.sdf_model.state_dict(),
            'current_iteration': self.current_iter,
        }
        torch.save(checkpoint, os.path.join(self.ckpt_path, 'ckpt_{:0>6d}.pth'.format(self.current_iter)))

if __name__ == '__main__':
    pretrain = Pretraining(conf)
    for cls in classes:
        pretrain_loader = get_pretrain_data_loader(cls, conf)
    pretrain.train(pretrain_loader)
    print("Finish Pretraining!")
            

    