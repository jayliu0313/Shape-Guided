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

################################### 指令 #################################

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--point_num', type=int, default=500)  # For one local region include how many points, and it would input to the model
parser.add_argument('--group_mul', type=int, default=5)
parser.add_argument('--sampled_size', type=int, default=20)
parser.add_argument('--grid_path', type=str, default = "grid_dir")
parser.add_argument('--ckpt_dir', type=str, default = "./checkpoint") # auto recreate if exist
CKPT_FILENAME = "knn500"
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument("--save_idx", type=int, default=-1)
parser.add_argument("--learning_rate", type=int, default=0.0001)
classes = [
"*"
]  # load category

a = parser.parse_args()
cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx
#torch.cuda.set_device(a.CUDA)

conf = Configuration(
    image_size = a.image_size,
    sampled_size = a.sampled_size,
    POINT_NUM = a.point_num,
    group_mul = a.group_mul,
    BS = a.batch_size,
    grid_path = a.grid_path,
    classes = classes,
    LR = a.learning_rate
)

# create the ckpt dir and conf file
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
conf.ckpt_dir = os.path.join(a.ckpt_dir, time) + CKPT_FILENAME
if not osp.exists(conf.ckpt_dir):
    os.makedirs(conf.ckpt_dir)
conf.save(os.path.join(conf.ckpt_dir, "Congiguration"))
RESULT_DIR = os.path.join(conf.ckpt_dir, "result")
os.makedirs(RESULT_DIR)

class Pretraining():
    def __init__(self, conf):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        # configuration
        self.BS = conf.BS
        self.POINT_NUM = conf.POINT_NUM
        self.ckpt_dir = conf.ckpt_dir
        learning_rate = conf.LR
        self.current_iter = 0

        # Initialize Model
        self.sdf_model = SDF_Model(self.POINT_NUM).to(self.device)
        self.optimizer = optim.Adam(self.sdf_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    def train(self, pretrain_loader):
        buf_size = 1  # Make 'training_stats' file to flush each output line regarding training.
        log_file = open(osp.join(self.ckpt_dir, "train_states.txt"), "a", buf_size)
        for epoch in range(1001):
            loss_list = []
            for points, samples in tqdm(pretrain_loader, desc=f'Pre-Training Epoch: {epoch}'):
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
                loss_list.append(loss)
               
            if(epoch % 200 == 0):
                for patch in range(self.BS):
                    gt_pc = point[patch].detach().cpu().numpy()
                    predict_pc = g_points[patch].detach().cpu().numpy()
                    save_pc(gt_pc, RESULT_DIR, str(epoch) + '_gt_patch' + str(patch))
                    save_pc(predict_pc, RESULT_DIR, str(epoch) + '_predict_patch' + str(patch))

            print('\nepoch: ', epoch, 'loss: ', sum(loss_list) / len(loss_list))
            if(epoch % 100 == 0):
                print('save model and epoch:', epoch ,'loss:',loss)
                self.save_checkpoint()
            if log_file is not None:
                log_file.write(
                    "epoch:%04d\tloss:%.9f\n" % (epoch, sum(loss_list)/len(loss_list))
                )
            self.current_iter += 1 
        log_file.close()
        self.save_checkpoint()

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.ckpt_dir, checkpoint_name), map_location=self.device)
        print(os.path.join(self.ckpt_dir, 'checkpoints', checkpoint_name))
        self.sdf_model.load_state_dict(checkpoint['sdf_model'])
        self.current_iter = checkpoint['current_iteration']
        
    def save_checkpoint(self):
        checkpoint = {
            'sdf_model': self.sdf_model.state_dict(),
            'current_iteration': self.current_iter,
        }
        torch.save(checkpoint, os.path.join(self.ckpt_dir, 'ckpt_{:0>6d}.pth'.format(self.current_iter)))

if __name__ == '__main__':
    pretrain = Pretraining(conf)
    for cls in classes:
        pretrain_loader = get_pretrain_data_loader(cls, conf)
    pretrain.train(pretrain_loader)
    print("Finish Pretraining!")
            

    