#-*-coding:utf-8-*-

import numpy as np
import xml.etree.ElementTree as ET
import SimpleITK as sitk
import find_scans as fs
from scipy.ndimage.morphology import binary_erosion
import math
import os
import cv2
import matplotlib.pyplot as plt
from skimage import measure, feature
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parseXML(scan_path):
    '''
    parse xml file

    args:
    xml file path

    output:
    nodule list
    [{nodule_id, roi:[{z, sop_uid, xy:[[x1,y1],[x2,y2],...]}]}]
    '''
    file_list = os.listdir(scan_path)
    for file in file_list:
        if file.split('.')[1] == 'xml':
            xml_file = file
            break
    prefix = "{http://www.nih.gov}"
    tree = ET.parse(scan_path +'/' + xml_file)
    root = tree.getroot()
    readingSession_list = root.findall(prefix + "readingSession") 
    nodules = []

    for session in readingSession_list:
        #print(session)
        unblinded_list = session.findall(prefix + "unblindedReadNodule")
        #print(unblinded_list)
        for unblinded in unblinded_list:
            nodule_id = unblinded.find(prefix + "noduleID").text
            edgeMap_num = len(unblinded.findall(prefix+"roi/"+prefix+"edgeMap"))
            if edgeMap_num > 1:
            # it's segmentation label
                nodule_info = {}
                nodule_info['nodule_id'] = nodule_id
                nodule_info['roi'] = []
                roi_list = unblinded.findall(prefix + "roi")
                for roi in roi_list:
                    roi_info = {}
                    roi_info['z'] = float(roi.find(prefix + "imageZposition").text)
                    roi_info['sop_uid'] = roi.find(prefix + "imageSOP_UID").text
                    roi_info['xy'] = []
                    edgeMap_list = roi.findall(prefix + "edgeMap")
                    for edgeMap in edgeMap_list:
                        x = float(edgeMap.find(prefix + "xCoord").text)
                        y = float(edgeMap.find(prefix + "yCoord").text)
                        xy = [x, y]
                        roi_info['xy'].append(xy)
                    nodule_info['roi'].append(roi_info)
                nodules.append(nodule_info)
    return nodules


def coord_trans(nodules, scan_path):
    '''
    transform z coord from world to voxel

    args:
    nodule: dict of nodule info, {nodule_id, roi:[{z, sop_uid, xy:[[x,y]]}]}
    scan_path

    output:
    nodules_boundary_coords:
    [{nodule_id, boundary_coords:[[x1,y1,z1],[x2,y2,z2],...]}]

    '''
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(scan_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)    #z, y, x
    origin = image.GetOrigin()         #x, y, z
    spacing = image.GetSpacing()       #x, y, z
    deps = image_array.shape[0]
    cols = image_array.shape[1]
    rows = image_array.shape[2]

    transed_nodule_list = []
    nodule_num = 0
    for nodule in nodules:
        #nodule_id = nodule['nodule_id']
        label_image = np.zeros((deps, cols, rows), dtype=int)  # the array saved binary label array
        for roi in nodule['roi']:
            roi['z'] = np.rint((roi['z'] - origin[2])/spacing[2])     # trans z from world to voxel
            for xy in roi['xy']:
                label_image[int(roi['z']), int(xy[1]), int(xy[0])] = 1  #boundary points in label image = 1
        boundary_coords = []
        index = np.where(label_image==1)   # find boundary coords
        for i in range(index[0].shape[0]):
            boundary_coords.append([index[0][i], index[1][i], index[2][i]])
        transed_nodule = {}
        transed_nodule['nodule_id'] = nodule_num
        transed_nodule['boundary_coords'] = boundary_coords
        transed_nodule_list.append(transed_nodule)
        nodule_num += 1

    return transed_nodule_list, image_array, deps, cols, rows

def bounding_box(transed_nodule_dict):
    boundary_arr = np.array(transed_nodule_dict['boundary_coords'])
    col_max = boundary_arr.max(axis=0)
    col_min = boundary_arr.min(axis=0)
    return col_max, col_min

def duplicate_nodules(transed_nodule_list):
    for i in range(len(transed_nodule_list)-1):
        for j in range(i+1, len(transed_nodule_list)):
            if transed_nodule_list[i]['nodule_id'] != transed_nodule_list[j]['nodule_id']:
                i_col_max, i_col_min = bounding_box(transed_nodule_list[i])
                j_col_max, j_col_min = bounding_box(transed_nodule_list[j])
                z_low = max(i_col_min[0], j_col_min[0])
                z_high = min(i_col_max[0], j_col_max[0])
                y_low = max(i_col_min[1], j_col_min[1])
                y_high = min(i_col_max[1], j_col_max[1])
                x_low = max(i_col_min[2], j_col_min[2])
                x_high = min(i_col_max[2], j_col_max[2])
                if z_low >= z_high or y_low >= y_high or x_low >= x_high:
                    iou = 0
                else:
                    inter_area = (z_high - z_low) * (y_high - y_low) * (x_high - x_low)
                    i_bbox_area = (i_col_max[0] - i_col_min[0]) * (i_col_max[1] - i_col_min[1]) * (i_col_max[2] - i_col_min[2])
                    j_bbox_area = (j_col_max[0] - j_col_min[0]) * (j_col_max[1] - j_col_min[1]) * (j_col_max[2] - j_col_min[2])
                    iou = inter_area / (i_bbox_area + j_bbox_area - inter_area)
                if iou >=0.4:
                    transed_nodule_list[j]['nodule_id'] = transed_nodule_list[i]['nodule_id']
    return transed_nodule_list

def fill_hole(transed_nodule_list, deps, cols, rows):
    filled_nodule_list = []

    for nodule_dict in transed_nodule_list:
        filled_nodule = {}
        filled_nodule['nodule_id'] = nodule_dict['nodule_id']
        filled_nodule['coords'] = []
        label_image = np.zeros((deps, cols, rows),  dtype=int)
        for coord in nodule_dict['boundary_coords']:
            label_image[coord[0], coord[1], coord[2]] = 1
        for i in range(deps):
            label_image[i,:,:] = fill_nodule(label_image[i,:,:])             # fill segmentation mask
            #label_image[i,:,:] = binary_erosion(label_image[i,:,:]).astype(label_image[i,:,:].dtype)
        index = np.where(label_image==1)
        for i in range(index[0].shape[0]):
            filled_nodule['coords'].append([index[0][i], index[1][i], index[2][i]])
        filled_nodule_list.append(filled_nodule)

    return filled_nodule_list



def fill_nodule(nodule_z):
    h, w = nodule_z.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = nodule_z.copy()
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return (~canvas | nodule_z.astype(np.uint8))


def calc_union_freq(filled_nodule_list):
    union_nodule_list = []
    nodule_id = []
    for i in range(len(filled_nodule_list)-1):
        if filled_nodule_list[i]['nodule_id'] not in nodule_id:
        # the nodule has not been calculate union yet
            nodule_id.append(filled_nodule_list[i]['nodule_id'])
            union = {}
            union['nodule_id'] = filled_nodule_list[i]['nodule_id']
            union['coords'] = []
            union['freq'] = []
            union['radiologist'] = 0
            for j in range(i, len(filled_nodule_list)):
            # because 'union' keys is all empty, so index 'j' should start at 'i'
                if filled_nodule_list[j]['nodule_id'] == union['nodule_id']:
                # they are the same nodule
                    for coord in filled_nodule_list[j]['coords']:
                        if coord not in union['coords']:
                            union['coords'].append(coord)
                            union['freq'].append(1)
                        else:
                            union['freq'][union['coords'].index(coord)] += 1
            
            union['radiologist'] = max(union['freq'])
            union['freq'] = [i/max(union['freq']) for i in union['freq']]
            

            union_nodule_list.append(union)

    return union_nodule_list

def plot_3d(image, lidc_id, nodule_id, pic_s_path, threshold=0.5):
    p = image.transpose(2,1,0)

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.savefig(pic_s_path+lidc_id+"_label_"+str(nodule_id)+".png")

def generate_cube(union_nodule_list, image_array, sample_s_path, label_s_path, pic_s_path, lidc_id):
    deps = image_array.shape[0]
    cols = image_array.shape[1]
    rows = image_array.shape[2]

    for nodule in union_nodule_list:
        nodule_id = nodule['nodule_id']
        roi = np.array(nodule['coords'])
        #print(roi.shape)
        score = np.array(nodule['freq'])
        #print(score.shape)
        
        col_max = roi.max(axis=0)    # zyx
        col_min = roi.min(axis=0)
        cube_d = col_max[0] - col_min[0] + 1
        cube_h = col_max[1] - col_min[1] + 1
        cube_w = col_max[2] - col_min[2] + 1
        label_array = np.zeros([cube_d, cube_h, cube_w])
        #print(label_array.shape)
        pic_array = np.zeros([cube_d, cube_h, cube_w])
        for i in range(roi.shape[0]):
            label_array[int(roi[i,0]-col_min[0]), int(roi[i,1]-col_min[1]), int(roi[i,2]-col_min[2])] = score[i]
            pic_array[int(roi[i,0]-col_min[0]), int(roi[i,1]-col_min[1]), int(roi[i,2]-col_min[2])] = 1
        nodule_array = image_array[col_min[0]:(col_max[0]+1), col_min[1]:(col_max[1]+1), col_min[2]:(col_max[2]+1)]
        assert nodule_array.shape == label_array.shape 
        if nodule_array.shape[0] > 2 and min(nodule_array.shape[1], nodule_array.shape[2]) >= 10 and nodule['radiologist'] == 4:
            plot_3d(pic_array, lidc_id, nodule_id, pic_s_path)
            np.save(sample_s_path + lidc_id + "_sample_" + str(nodule_id) + ".npy", nodule_array)
            np.save(label_s_path + lidc_id + "_label_" + str(nodule_id) + ".npy", label_array)
            print("Saved %s's no.%d nodule" % (lidc_id, nodule_id))



if __name__ == '__main__':
    scan_path = "D:/DOI/data/LIDC-IDRI-0003/1.3.6.1.4.1.14519.5.2.1.6279.6001.101370605276577556143013894866/1.3.6.1.4.1.14519.5.2.1.6279.6001.170706757615202213033480003264"
    lidc_path = "D:/DOI/"
    out_path = lidc_path + "seg_3d_samples/"
    data_path = lidc_path + "data/"
    label_path = out_path + "label/"
    sample_path = out_path + "sample/"
    pic_path = out_path + "picture/"
    nodules = parseXML(scan_path)
    #print(nodules)
    transed_nodule_list, image_array, deps, cols, rows = coord_trans(nodules, scan_path)
    #print(transed_nodule_list)
    transed_nodule_list = duplicate_nodules(transed_nodule_list)
    #print(transed_nodule_list)
    filled_nodule_list = fill_hole(transed_nodule_list, deps, cols, rows)
    union_nodule_list = calc_union_freq(filled_nodule_list)
    #for i in range(len(union_nodule_list)):
        #print(union_nodule_list[i]['radiologist'])
    #generate_cube(union_nodule_list, image_array, sample_path, label_path, pic_path, "LIDC-IDRI-0003")
    
