#-*-coding:utf-8-*-


import numpy as np
import os


def read_patch(sample_file, config):

    sample_arr = np.load(os.path.join(config['train_data_dir'] + "sample/", sample_file))
    lidc_id = sample_file.split("_")[0]
    nodule_id = sample_file.split("_")[2]
    label_file = lidc_id + "_label_" + nodule_id
    seg_arr = np.load(os.path.join(config['train_data_dir'] + "label/", label_file))
    label = np.zeros((seg_arr.shape[0], seg_arr.shape[1], seg_arr.shape[2], 2), dtype=np.float32)
    label[seg_arr == 0, 0] = 1
    label[seg_arr == 1, 1] = 1

    sample_arr = np.expand_dims(sample_arr, 3)

    assert sample_arr.shape == (32, 64, 64, 1) and label.shape == (32, 64, 64, 2)
    return sample_arr, label

def init_read(sample_file, label_file):

    sample_arr = np.load(sample_file)
    seg_arr = np.load(label_file)
    label = np.zeros((seg_arr.shape[0], seg_arr.shape[1], seg_arr.shape[2], 2), dtype=np.float32)
    label[seg_arr == 0, 0] = 1
    label[seg_arr == 1, 1] = 1

    sample_arr = np.expand_dims(sample_arr, 3)

    assert sample_arr.shape == (32, 64, 64, 1) and label.shape == (32, 64, 64, 2)
    return sample_arr, label
