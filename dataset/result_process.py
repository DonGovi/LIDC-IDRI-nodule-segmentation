#-*-coding:utf-8-*-

import numpy as np
import os
import data_process as dp
from plot_nodule import plot_arr
import matplotlib.pyplot as plt

def merge_result(result_file, result_path):
    result_arr = np.load(os.path.join(result_path, result_file))
    #print(result_arr.shape)
    label_arr = np.zeros((result_arr.shape[1], result_arr.shape[2], result_arr.shape[3]), dtype=np.float32)
    result_arr = result_arr[0, ...]
    for z in range(label_arr.shape[0]):
        for y in range(label_arr.shape[1]):
            for x in range(label_arr.shape[2]):
                if result_arr[z,y,x,0] >= result_arr[z,y,x,1]:
                    label_arr[z,y,x] = 0
                else:
                    label_arr[z,y,x] = 1

    return label_arr





if __name__ == '__main__':
    result_path = "E:/lidc_seg/lidc_padding_samples/test/label/"
    #label_path = "E:/lidc_seg/lidc_padding_samples/test/label/LIDC-IDRI-0004_label_0.npy"
    save_path = "E:/lidc_seg/lidc_padding_samples/test/label_picture/"
    result_list = os.listdir(result_path)
    for result_file in result_list:
        print(result_file)
        #result_arr = merge_result(result_file, result_path)
        result_arr = np.load(result_path+result_file)
        result_name = result_file.split(".")[0]
        save_name = save_path + result_name + ".png"
        print(save_name)
        plot_arr(result_arr, save_name)
        
