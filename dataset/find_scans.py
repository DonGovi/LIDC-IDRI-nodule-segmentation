#-*-coding:utf-8-*-

import os


def dcm_count(file_list):
    dcm_count = 0
    for file_name in file_list:
        if file_name.split('.')[1] == 'dcm':
            dcm_count += 1
    return dcm_count

def find_scan(path):
    path_list = []   #paths that contain .dcm files
    dcm_counts = []   #num of .dcm files

    subdir_list = os.listdir(path)
    for subdir in subdir_list:
        subdir = os.path.join(path, subdir)
        subsubdir_list = os.listdir(subdir)
        for subsubdir in subsubdir_list:
            subsubdir = os.path.join(subdir, subsubdir)
            file_list = os.listdir(subsubdir)
            #print(file_list)
            count = dcm_count(file_list)
            path_list.append(subsubdir)
            dcm_counts.append(count)

    return path_list[dcm_counts.index(max(dcm_counts))]












