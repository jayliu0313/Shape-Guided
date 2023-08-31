import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import os.path as osp

model_path = ""
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
absolute_path = os.path.join(parent_dir, model_path)
def main():
    #visualize training loss of AE
    input_path = os.path.join(absolute_path, "train_states.txt")
    save_path = os.path.join(absolute_path, "loss_info.png")
    visualize_loss(input_path, save_path) 

def visualize_loss(path, save_path):
    pretrained_info = {
        "epoch": [],
        "loss": [],
    } 
    with open(path) as f:
        for line in f.readlines():
            s = line.split('\t')
            epoch = s[0].split(':')
            loss = s[1].split(':')
            pretrained_info['epoch'].append(epoch[1])
            pretrained_info['loss'].append(float(loss[1]))
    
    step_df = pd.DataFrame(pretrained_info)

    plt.title('Pre_trained loss detail') # set the title of graph
    plt.figure(figsize=(10, 7))
    axes = plt.axes()
    axes.set_ylim([0.012, 0.016]) # set the range of y value
    plt.plot(step_df['epoch'], step_df['loss'], color='b')
    plt.xticks(np.arange(0, len(step_df['epoch'])+1, 80))
    plt.xlabel('epoch') # set the title of x axis
    plt.ylabel('loss')
    plt.savefig(save_path)

if __name__ == '__main__':
    main()