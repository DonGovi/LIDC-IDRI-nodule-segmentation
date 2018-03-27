#-*-coding:utf-8-*-

import numpy as np
import os
from data_padding import padding_patch, normalization
from plot_nodule import plot_arr

lidc_path = "D:/DOI/"
origin_path = os.path.join(lidc_path, "four_radiologist_samples/")
padding_path = os.path.join(lidc_path, "lidc_padding_samples/")
origin_sample_path = os.path.join(origin_path, "sample/")
origin_label_path = os.path.join(origin_path, "label/")
padding_sample_path = os.path.join(padding_path, "sample/")
padding_label_path = os.path.join(padding_path, "label/")

nodule_file = os.path.join(origin_sample_path, "LIDC-IDRI-0004_sample_0.npy")
label_file = os.path.join(origin_label_path, "LIDC-IDRI-0004_label_0.npy")

nodule_arr = np.load(nodule_file)
label_arr = np.load(label_file)

nodule_64_arr, label_64_arr = padding_patch(nodule_arr, label_arr)
print(np.max(label_64_arr), np.min(label_64_arr))
print(np.max(label_arr), np.min(label_arr))

plot_arr(label_64_arr)