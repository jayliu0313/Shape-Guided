import argparse
import os
import os.path as osp
import datetime
import pandas as pd

from core.shape_guide_core import ShapeGuide, Configuration
from utils.utils import list_join


parser = argparse.ArgumentParser()
# model setup
parser.add_argument('--image_size', type=int, default=224, help="Reduced 800*800 image size to n*n by interpolate")
parser.add_argument('--point_num', type=int, default=500, help="The Number of pc for each 3D patch")

FILENAME = "-result"
# path setup
parser.add_argument('--datasets_path', type=str, default="<dataset_dir>", help="The dir path of mvtec3D-AD dataset")
parser.add_argument('--grid_path', type=str, default="<grid_dir>", help="The dir path of grid you cut, it would include training npz, testing npz")
parser.add_argument('--ckpt_path', type=str, default="checkpoint/best_ckpt/ckpt_000601.pth")      #It would load prtraining of ckpt
parser.add_argument('--output_dir', type=str, default='output/', help="The dir path of output")

# others
parser.add_argument('--CUDA', type=int, default=0, help="Choose the device of CUDA")
parser.add_argument('--viz', action="store_true", default=False, help="Visualize results with heatmap")
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

# It's changes will not affect training and testing
parser.add_argument('--group_mul', type=int, default=10, help="The group_mul multiplied by point_num is the number of groups")
parser.add_argument('--sampled_size', type=int, default=20, help="The Number of sampled queary point for pretrain")

a = parser.parse_args()
cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx

# configuration
conf = Configuration()
conf.image_size = a.image_size
conf.sampled_size = a.sampled_size
conf.BS = 1
conf.POINT_NUM = a.point_num
conf.group_mul = a.group_mul
conf.datasets_path = a.datasets_path
conf.grid_path = a.grid_path
conf.ckpt_path = a.ckpt_path
conf.classes = class_name
conf.method_name = ['RGB', 'SDF', 'RGB_SDF']         # RGB or SDF or RGBSDF
conf.rgb_method = 'Dict'
conf.k_number = 10
conf.dict_n_component = 3
conf.viz = a.viz

# create output dir and log file
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

conf.output_dir = os.path.join(a.output_dir, time)

conf.output_dir = conf.output_dir + FILENAME
if not osp.exists(conf.output_dir):
    os.makedirs(conf.output_dir)
conf.save(os.path.join(conf.output_dir, "Congiguration"))

# Setup Integration Limit
PRO_LIMIT = [0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.01]
METHOD_NAMES = conf.method_name
METHOD_NAMES_PRO = list_join(METHOD_NAMES, [str(x) for x in PRO_LIMIT])


if __name__ == "__main__":

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES_PRO, columns=['Method'])
    result_file = open(osp.join(conf.output_dir, "results.txt"), "a", 1)

    for cls in class_name:
        print(f"\nRunning on class {cls}\n")
        patchcore = ShapeGuide(conf, cls, PRO_LIMIT)
        patchcore.fit()
        patchcore.align()
        image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate()
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)

        print(f"\nFinished running on class {cls}")
        print("################################################################################\n")

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)

    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))
    result_file.write(f'Image ROCAUC Results \n\n{image_rocaucs_df.to_markdown(index=False)} \n\n')

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))
    result_file.write(f'Pixel ROCAUC Results \n\n{pixel_rocaucs_df.to_markdown(index=False)} \n\n')

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))
    result_file.write(f'AU PRO Results \n\n{au_pros_df.to_markdown(index=False)}')
    result_file.close()
