#-*-coding:utf-8-*-

import os
import numpy as np
import shutil

def seg_test(train_dir, test_dir, test_num=60):

    nodule_list = np.random.permutation(os.listdir(train_dir + "sample/"))

    ixs = np.random.choice(range(len(nodule_list)), size=test_num, replace=False)
    for ix in ixs:
        nodule_file = nodule_list[ix]
        lidc_id = nodule_file.split("_")[0]
        noudle_id = nodule_file.split("_")[2]
        label_file = lidc_id + "_label_" + noudle_id
        #picture_file = lidc_id + "_label_" + noudle_id.split(".")[0] + '.png'

        shutil.move(os.path.join(train_dir+'sample/', nodule_file), test_dir+"sample/")
        shutil.move(os.path.join(train_dir+'label/', label_file), test_dir+"label/")
        #shutil.move(os.path.join(train_dir+'picture/', picture_file), test_dir+"label_picture/")


if __name__ == '__main__':
    train_dir = "E:/lidc_seg/lidc_padding_samples/train/"
    test_dir = "E:/lidc_seg/lidc_padding_samples/test/"

    seg_test(train_dir, test_dir)