#-*-coding:utf-8-*-


import numpy as np
import matplotlib.pyplot as plt
import os


def plot_arr(array, save_name):
    #array[array>=threhold] = 1
    #array[array<threhold] = 0
    fig, axes = plt.subplots(4, 8, figsize=(64, 64))
    count = 0
    for i in range(4):
        for j in range(8):
            axes[i,j].imshow(array[count,...], cmap="gray")
            count += 1
    #fig.tight_layout()
    plt.savefig(save_name)

if __name__ == '__main__':
    
    data_path = "D:/DOI/seg_3d_samples/sample/"
    label_path = "D:/DOI/seg_3d_samples/label/"


    #nodule = data_path + "LIDC-IDRI-0016_sample_0.npy"
    nodule = label_path + 'LIDC-IDRI-0004_label_0.npy'
    label_list = os.listdir(label_path)

    for label in label_list:
        label_arr = np.load(label_path + label)
        if label_arr.shape[1] >= 30 or label_arr.shape[2] >= 30:
            label_arr[np.where(label_arr > 0)] = 1
            print(label)
            break

    nodule_arr = np.load(nodule)
    print(nodule_arr.shape)


    plt.subplots(1,1,figsize=(0.2,0.2))
    plt.imshow(nodule_arr[2,:,:], cmap='gray')
    plt.show()
