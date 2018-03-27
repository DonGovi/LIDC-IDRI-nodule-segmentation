#-*-coding:utf-8-*-

import data_process as dp
import numpy as np
import os
import find_scans as fs


lidc_path = "D:/DOI/"
out_path = lidc_path + "four_radiologist_samples/"
data_path = lidc_path + "data/"
label_path = out_path + "label/"
sample_path = out_path + "sample/"
pic_path = out_path + "picture/"

def sample_lidc(lidc_id):
    scan_path = fs.find_scan(data_path + lidc_id)
    nodules = dp.parseXML(scan_path)
    transed_nodule_list, image_array, deps, cols, rows = dp.coord_trans(nodules, scan_path)
    dup_nodule_list = dp.duplicate_nodules(transed_nodule_list)
    filled_nodule_list = dp.fill_hole(dup_nodule_list, deps, cols, rows)
    union_nodule_list = dp.calc_union_freq(filled_nodule_list)
    dp.generate_cube(union_nodule_list, image_array, sample_path, label_path, pic_path, lidc_id)


if __name__ == '__main__':

    lidc_list = os.listdir(data_path)
    for lidc_dir in lidc_list:
        print("Strating sample %s" % lidc_dir)
        sample_lidc(lidc_dir)

