import os
import os.path as osp
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from skimage import morphology
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
OUT_DIR = 'Visualization'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL = 'PointNet'

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 4/2.54 # New
dpi = 300   # New

def denormalization(x, norm_mean, norm_std):
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def export_test_images(test_img, gts, scores, threshold, output_dir):
    
    image_dirs = os.path.join(output_dir, OUT_DIR ,'images')
    
    if not os.path.isdir(image_dirs):
        print('Exporting images...')
        os.makedirs(image_dirs, exist_ok=True)

        kernel = morphology.disk(2)
        scores_norm = 1.0/scores.max()

        for i in tqdm(range(0, len(test_img), 1), desc="export heat map image"):
            img = test_img[i]
            img = denormalization(img, IMAGENET_MEAN, IMAGENET_STD)
            
            # gts
            gt_mask = gts[i].astype(np.float64)
            gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')

            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            #
            fig_img, ax_img = plt.subplots(3, 1, figsize=(2*cm, 6*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0].imshow(gt_img)
            ax_img[1].imshow(img, cmap='gray', interpolation='none')
            ax_img[1].imshow(score_map, cmap='jet', norm=norm, alpha=0.5, interpolation='none')
            ax_img[2].imshow(score_img)
            image_file = os.path.join(image_dirs, '{:08d}'.format(i) + '.png')
            fig_img.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()

def visualization(test_image_list, gt_label, score_label, gt_mask_list, super_mask_list, output_dir):
    
    gt_mask = np.asarray(gt_mask_list)
    super_mask = np.asarray(super_mask_list)
    
    precision, recall, thresholds = precision_recall_curve(gt_label, score_label)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    det_threshold = thresholds[np.argmax(f1)]

    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), super_mask.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    seg_threshold = thresholds[np.argmax(f1)]

    export_test_images(test_image_list, gt_mask, super_mask, seg_threshold, output_dir)
    return det_threshold, seg_threshold

def visualize_image_s_distribute(sdf_s, rgb_s, image_gt, output_dir):
    path_dirs = os.path.join(output_dir, OUT_DIR)
    os.makedirs(path_dirs, exist_ok=True) 
    image_file = os.path.join(path_dirs, 'image_score_dis.png')
    
    image_gt = image_gt.reshape(-1)
    colors = np.array(["blue", "red"])
    com_s = sdf_s * rgb_s
    x = range(len(sdf_s))

    fig = plt.figure(figsize=(12, 10)) 
    plt.subplots_adjust(
                    bottom=0.1, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    
    fig.text(0.5, 0.04, s = 'Image ID', ha='center', fontsize=20)
    fig.text(0.04, 0.4, s = 'Image-level Score', ha='center', rotation='vertical', fontsize=20)

    ax = fig.add_subplot(311)
    ax.set_title("SDF image score Distribution", fontsize=18)
    ax.scatter(x, sdf_s, c=colors[image_gt], s=50)
    ax.plot(x, sdf_s)
    
    ax = fig.add_subplot(312)
    ax.set_title("RGB image score Distribution", fontsize=18)
    ax.scatter(x, rgb_s, c=colors[image_gt], s=50)
    ax.plot(x, rgb_s)

    ax = fig.add_subplot(313)
    ax.set_title("Shape-Guided image score Distribution", fontsize=18)
    ax.scatter(x, com_s, c=colors[image_gt], s=50)
    ax.plot(x, com_s)
    # plt.legend(loc='best')
    plt.savefig(image_file)
    plt.close()

def visualize_smap_distribute(total_map, sdf_map, rgb_map, new_rgb_map, image_size, output_dir):
    path_dirs = os.path.join(output_dir, OUT_DIR, "smap_ditribute")
    os.makedirs(path_dirs, exist_ok=True)
    image_file = os.path.join(path_dirs, 'category_all_ditribution.png')

    non_min_total_map = total_map[total_map > total_map.min()]
    non_min_sdf_map = sdf_map[sdf_map > sdf_map.min()]
    non_min_rgb_map = rgb_map[rgb_map > rgb_map.min()]
    non_min_new_rgb = new_rgb_map[new_rgb_map > new_rgb_map.min()]

    fig = plt.figure(figsize=(14, 16))
    fig.text(0.5, 0.04, s = 'pixel-level score', ha='center', fontsize=20)
    fig.text(0.04, 0.45, s = 'number of pixel', ha='center', rotation='vertical', fontsize=20)
    ax = fig.add_subplot(411)
    ax.hist(non_min_total_map, bins=100, color='r')
    ax.title.set_text("Total Distribution")
        
    ax = fig.add_subplot(412)
    ax.hist(non_min_sdf_map, bins=100, color='g')
    ax.title.set_text("SDF Distribution")

    ax = fig.add_subplot(413)
    ax.hist(non_min_rgb_map, bins=100, color='b')
    ax.title.set_text("RGB Distribution")

    ax = fig.add_subplot(414)
    ax.hist(non_min_new_rgb, bins=100, color='c')
    ax.title.set_text("New RGB Distribution")

    plt.savefig(image_file)
    plt.close()

    total_map = total_map.reshape(-1, image_size * image_size)
    sdf_map = sdf_map.reshape(-1, image_size * image_size)
    rgb_map = rgb_map.reshape(-1, image_size * image_size)
    new_rgb_map = new_rgb_map.reshape(-1, image_size * image_size)
    for i in tqdm(range(0, len(total_map), 5), desc='export score map distribution'):
        image_file = os.path.join(path_dirs, '{:08d}'.format(i) + '.png')
        non_min_total_map = total_map[i][total_map[i] > total_map[i].min()]
        non_min_sdf_map = sdf_map[i][sdf_map[i] > sdf_map[i].min()]
        non_min_rgb_map = rgb_map[i][rgb_map[i] > rgb_map[i].min()]
        non_min_new_rgb = new_rgb_map[i][new_rgb_map[i] > new_rgb_map[i].min()]
        
        fig = plt.figure(figsize=(14, 16))
        fig.text(0.5, 0.04, s = 'pixel-level score', ha='center', fontsize=20)
        fig.text(0.04, 0.45, s = 'number of pixel', ha='center', rotation='vertical', fontsize=20)

        ax1 = fig.add_subplot(411)
        ax1.hist(non_min_total_map, bins=100, color='r')
        ax1.title.set_text("Total Distribution")
        
        ax1 = fig.add_subplot(412)
        ax1.hist(non_min_sdf_map, bins=100, color='g')
        ax1.title.set_text("SDF Distribution")

        ax1 = fig.add_subplot(413)
        ax1.hist(non_min_rgb_map, bins=100, color='b')
        ax1.title.set_text("Original RGB Distribution")

        ax1 = fig.add_subplot(414)
        ax1.hist(non_min_new_rgb, bins=100, color='c')
        ax1.title.set_text("Adjust RGB Distribution")
        
        plt.savefig(image_file)
        plt.close()