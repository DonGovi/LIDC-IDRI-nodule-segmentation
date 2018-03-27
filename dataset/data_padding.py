#-*-coding:utf-8-*-

import numpy as np
import os
import math


def padding_patch(nodule_arr, label_arr, cube_size_xy=64, cube_size_z=32):
    nodule_64_arr = np.ones((cube_size_z, cube_size_xy, cube_size_xy), dtype=np.float32)
    nodule_64_arr *= -1000.0
    label_64_arr = np.zeros((cube_size_z, cube_size_xy, cube_size_xy), dtype=np.float32)
    #nodule_arr = normalization(nodule_arr)

    origin_shape = nodule_arr.shape
    padding_z = (cube_size_z - origin_shape[0])/2
    padding_y = (cube_size_xy - origin_shape[1])/2
    padding_x = (cube_size_xy - origin_shape[2])/2

    edge_z_min = math.floor(padding_z)
    edge_z_max = edge_z_min + origin_shape[0]
    edge_y_min = math.floor(padding_y)
    edge_y_max = edge_y_min + origin_shape[1]
    edge_x_min = math.floor(padding_x)
    edge_x_max = edge_x_min + origin_shape[2]
    print([edge_z_min, edge_z_max, edge_y_min, edge_y_max, edge_x_min, edge_x_max])

    nodule_64_arr[edge_z_min:edge_z_max, edge_y_min:edge_y_max, edge_x_min:edge_x_max] = nodule_arr
    label_64_arr[edge_z_min:edge_z_max, edge_y_min:edge_y_max, edge_x_min:edge_x_max] = label_arr

    return nodule_64_arr, label_64_arr


def normalization(nodule_arr, max_val = 600.0, min_val = -1000.0):
    nodule_arr = nodule_arr.astype("float32")
    nodule_arr = (nodule_arr - min_val) / (max_val - min_val)
    nodule_arr = np.clip(nodule_arr, 0, 1)

    return nodule_arr


if __name__ == '__main__':
    lidc_path = "E:/LIDC-IDRI/"
    origin_path = os.path.join(lidc_path, "four_radiologist_samples/")
    padding_path = os.path.join(lidc_path, "lidc_padding_samples/")
    origin_sample_path = os.path.join(origin_path, "sample/")
    origin_label_path = os.path.join(origin_path, "label/")
    padding_sample_path = os.path.join(padding_path, "sample/")
    padding_label_path = os.path.join(padding_path, "label/")

    nodule_list = os.listdir(origin_sample_path)
    for nodule_file in nodule_list:
        lidc_id = nodule_file.split("_")[0]
        nodule_id = nodule_file.split("_")[2]
        label_file = lidc_id + "_label_" + nodule_id
        nodule_arr = np.load(os.path.join(origin_sample_path, nodule_file))
        label_arr = np.load(os.path.join(origin_label_path, label_file))
        label_arr = label_arr.astype("float32")
        if nodule_arr.shape[0] <=32 and max(nodule_arr.shape[1], nodule_arr.shape[2]) <= 64:
            nodule_64_arr, label_64_arr = padding_patch(nodule_arr, label_arr)
            assert nodule_64_arr.shape == label_64_arr.shape
            #print(nodule_64_arr.shape)
            np.save(os.path.join(padding_sample_path, nodule_file), nodule_64_arr)
            np.save(os.path.join(padding_label_path, label_file), label_64_arr)

