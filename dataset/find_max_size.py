#-*-coding:utf-8-*-

import numpy as np
import os

nodule_path = "D:/DOI/seg_sample/sample/"

nodule_list = os.listdir(nodule_path)

max_xy = 0
max_xy_nodule = "test.npy"
max_z = 0
max_z_nodule = "test.npy"
min_xy = 512

count = 0
big_nodule_list = []

for nodule in nodule_list:
    nodule_arr = np.load(nodule_path + nodule)
    if max(nodule_arr.shape[1], nodule_arr.shape[2]) > max_xy:
        max_xy = max(nodule_arr.shape[1], nodule_arr.shape[2])
        max_xy_nodule = nodule
    if nodule_arr.shape[0] > max_z:
        max_z = nodule_arr.shape[0]
        max_z_nodule = nodule
    if min(nodule_arr.shape[1], nodule_arr.shape[2]) < min_xy:
        min_xy = min(nodule_arr.shape[1], nodule_arr.shape[2])
    if max(nodule_arr.shape[1], nodule_arr.shape[2]) > 64:
        count += 1
        big_nodule_list.append(nodule)


print(max_xy, max_xy_nodule)
print(max_z, max_z_nodule)
print(count)
print(big_nodule_list)