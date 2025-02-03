import os
from os.path import dirname, realpath, join
import sys
from sys import argv, exit
from pathlib import Path
import nrrd
import pickle
import csv
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from .uuuuuu import make_contour_img, make_atlas_label_contour
from .obj_items import render_volume, render_small_volume


def _make_label_info_data_waxholm_rat(label_file_path, excel_file_path):
    # label_file_path = '..../WHS_SD_rat_atlas_v4.label'
    # excel_file_path = '..../WHS SD rat brain atlas v4 labels for MBAT.xlsx'

    xl_file = pd.ExcelFile(excel_file_path)
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    dfs_keys = list(dfs.keys())
    if len(dfs_keys) != 1:
        raise Exception('need to be only 1 sheet')

    df = dfs[dfs_keys[0]]

    index = []
    level = []
    name = []

    for i in range(df.shape[0]):
        da_line = df.iloc[i].values
        for j in range(df.shape[1]):
            if ~np.isnan(da_line[j]):
                print(da_line[j])
                index.append(da_line[j])
                level.append(j)
                name.append(da_line[j+1])
                break

    index = np.ravel(index).astype(int)
    abv = df['Abbreviation'].values[:len(index)]
    parent = df['Parent'].values[:len(index)].astype(int)

    file = open(label_file_path, 'rb')
    lines = file.readlines()
    file.close()

    for i in range(len(lines)):
        da_line = lines[i].decode()
        if da_line[0] == '#':
            continue
        start_line = i
        break

    lindex = []
    red = []
    green = []
    blue = []
    lname = []
    for i in range(start_line, len(lines)):
        da_line = lines[i].decode()
        print(da_line)
        da_elements = da_line.split('"')
        da_numbers = da_elements[0].split()
        lindex.append(int(da_numbers[0]))
        red.append(int(da_numbers[1]))
        green.append(int(da_numbers[2]))
        blue.append(int(da_numbers[3]))
        lname.append(da_elements[1])

    lindex = np.ravel(lindex)
    red = np.ravel(red)
    green = np.ravel(green)
    blue = np.ravel(blue)

    colors = np.zeros((len(index), 3))
    colors[:] = np.nan

    for i in range(1, len(lname)):
        print(i)
        if lindex[i] not in index:
            raise Exception('not matching')

    for i in range(len(index)):
        if index[i] > 600:
            if index[i] == 1000:
                colors[i] = np.array([50, 168, 82])
            elif index[i] in [1001, 1050, 1002, 1003, 1004, 1005, 1006, 1051, 1007, 1008, 1009, 1010, 1011, 1012]:
                colors[i] = np.array([255, 255, 255])
            elif index[i] == 1048:
                colors[i] = np.array([114, 126, 186])
            elif index[i] == 1049:
                colors[i] = np.array([16, 79, 24])
            else:
                colors[i] = np.array([128, 128, 128])
        else:
            da_ind = np.where(lindex == index[i])[0][0]
            colors[i] = np.array([red[da_ind], green[da_ind], blue[da_ind]])

    label = {}
    label['index'] = index
    label['color'] = colors.astype(int)
    label['label'] = name
    label['abbrev'] = abv
    label['parent'] = parent
    label['level_indicator'] = np.ravel(level)

    outfile = open('atlas_labels.pkl', 'wb')
    pickle.dump(label, outfile)
    outfile.close()


def check_data_path_and_load(file_path):
    success = True
    filename, file_extension = os.path.splitext(file_path)
    if file_extension == '.pkl':
        try:
            infile = open(file_path, 'rb')
            data = pickle.load(infile)
            infile.close()
        except (IndexError, AttributeError, IOError, OSError):
            success = False
    elif file_extension == '.nrrd':
        try:
            data, header = nrrd.read(file_path)
        except (IndexError, AttributeError, IOError, OSError):
            success = False
    else:
        try:
            data_file = nib.load(file_path)
            data = data_file.get_fdata()
        except (IndexError, AttributeError, IOError, OSError):
            success = False
    return data, success



def check_atlas_file_path(atlas_folder, data_file=None, segmentation_file=None):
    atlas_path = os.path.join(atlas_folder, data_file)
    segmentation_path = os.path.join(atlas_folder, segmentation_file)

    if not os.path.exists(atlas_path) or not os.path.exists(segmentation_path):
        msg = 'atlas_path or segmentation_path not exist.'
        msg_flag = 0
    else:
        msg = 'atlas path and segmentation path exist.'
        msg_flag = 1

    return msg, msg_flag, atlas_path, segmentation_path


def load_mask_file(atlas_folder, mask_file=None):
    if mask_file is not None:
        mask_path = os.path.join(atlas_folder, mask_file)
        if os.path.exists(mask_path):
            mask_data, mask_success = check_data_path_and_load(mask_path)
            if not mask_success:
                msg = 'Failed to load mask data.'
                msg_flag = 0
                mask_data = None
            else:
                msg = 'Mask data loaded successfully.'
                msg_flag = 1
                mask_data = mask_data[:, :, :, 0]  # ????? check data again when you have time, dear
        else:
            msg = 'Mask path not exist.'
            msg_flag = 0
            mask_data = None
    else:
        msg = 'No mask data is needed.'
        msg_flag = 1
        mask_data = None

    return msg, msg_flag, mask_data


def process_segmentation_data(atlas_folder, segmentation_path, mask_data):
    segmentation_data, seg_success = check_data_path_and_load(segmentation_path)
    if not seg_success:
        msg = 'Failed to load segmentation data.'
        msg_flag = 0
    else:
        if mask_data is not None:
            # make segmentation with mask
            for i in range(len(mask_data)):
                segmentation_data[i][mask_data[i] == 0] = 0
            segmentation_data = segmentation_data.astype('int')

        unique_label = np.unique(segmentation_data)

        segment = {'data': segmentation_data, 'unique_label': unique_label}

        outfile = open(os.path.join(atlas_folder, 'segment_pre_made.pkl'), 'wb')
        pickle.dump(segment, outfile)
        outfile.close()

        msg = 'Segmentation data processed successfully.'
        msg_flag = 1

    return msg, msg_flag


def process_atlas_data(atlas_folder, atlas_path, mask_data,
                       bregma_coordinates=None, lambda_coordinates=None, voxel_size=None):
    atlas_data, atlas_success = check_data_path_and_load(atlas_path)
    if not atlas_success:
        msg = 'Failed to load atlas volume data.'
        msg_flag = 0
    else:

        new_atlas_data = atlas_data.copy()
        if mask_data is not None:
            # make atlas with mask
            new_atlas_data = new_atlas_data - np.min(new_atlas_data)
            for i in range(len(mask_data)):
                new_atlas_data[i][mask_data[i] == 0] = 0
            new_atlas_data = new_atlas_data / np.max(new_atlas_data)
        else:
            new_atlas_data = new_atlas_data - np.min(new_atlas_data)
            new_atlas_data = new_atlas_data / np.max(new_atlas_data)

        atlas_info = [
            {'name': 'anterior', 'values': np.arange(new_atlas_data.shape[0]) * voxel_size, 'units': 'um'},
            {'name': 'dorsal', 'values': np.arange(new_atlas_data.shape[1]) * voxel_size, 'units': 'um'},
            {'name': 'right', 'values': np.arange(new_atlas_data.shape[2]) * voxel_size, 'units': 'um'},
            {'vxsize': voxel_size,
             'Bregma': [bregma_coordinates[0], bregma_coordinates[1], bregma_coordinates[2]],
             'Lambda': [lambda_coordinates[0], lambda_coordinates[1], lambda_coordinates[2]]}
        ]

        atlas = {'data': new_atlas_data, 'info': atlas_info}

        atlas_data = atlas['data']
        atlas_info = atlas['info']

        outfile = open(os.path.join(atlas_folder, 'atlas_pre_made.pkl'), 'wb')
        pickle.dump(atlas, outfile)
        outfile.close()

        msg = 'Volume Atlas data processed successfully.'
        msg_flag = 1

    return msg, msg_flag


def process_contour_data(segmentation_data, dim_index=0):
    contour_img = np.zeros(segmentation_data.shape, 'i')

    # pre-process boundary
    if dim_index == 0:
        for i in range(segmentation_data.shape[dim_index]):
            da_slice = segmentation_data[i, :, :].copy()
            da_contour = make_contour_img(da_slice)
            contour_img[i, :, :] = da_contour
    elif dim_index == 1:
        for i in range(segmentation_data.shape[dim_index]):
            da_slice = segmentation_data[:, i, :].copy()
            da_contour = make_contour_img(da_slice)
            contour_img[:, i, :] = da_contour
    else:
        for i in range(segmentation_data.shape[dim_index]):
            da_slice = segmentation_data[:, :, i].copy()
            da_contour = make_contour_img(da_slice)
            contour_img[:, :, i] = da_contour
    return contour_img



# boundary = {'s_contour': sagital_contour_img,
#                 'c_contour': coronal_contour_img,
#                 'h_contour': horizontal_contour_img}
#
#     bnd = {'data': boundary}
#
#     outfile_ct = open(os.path.join(atlas_folder, 'contour_pre_made.pkl'), 'wb')
#     pickle.dump(bnd, outfile_ct)
#     outfile_ct.close()

def process_atlas_raw_data(atlas_folder, data_file=None, segmentation_file=None, mask_file=None,
                           bregma_coordinates=None, lambda_coordinates=None, voxel_size=None):
    atlas_data, atlas_info, segmentation_data, boundary = None, None, None, None

    atlas_path = os.path.join(atlas_folder, data_file)
    segmentation_path = os.path.join(atlas_folder, segmentation_file)

    if mask_file is not None:
        mask_path = os.path.join(atlas_folder, mask_file)
        if os.path.exists(mask_path):
            mask_data, mask_success = check_data_path_and_load(mask_path)
            if not mask_success:
                msg = 'Failed to load mask data.'
                return msg
            mask_data = mask_data[:, :, :, 0]  # ????? check data again when you have time, dear
        else:
            mask_data = None
    else:
        mask_data = None

    if not os.path.exists(atlas_path) or not os.path.exists(segmentation_path):
        msg = 'atlas_path or segmentation_path not exist.'
        return msg

    # pre-process segmentation
    segmentation_data, seg_success = check_data_path_and_load(segmentation_path)
    if not seg_success:
        msg = 'Failed to load segmentation data.'
        return msg

    if mask_data is not None:
        # make segmentation with mask
        for i in range(len(mask_data)):
            segmentation_data[i][mask_data[i] == 0] = 0
        segmentation_data = segmentation_data.astype('int')

    unique_label = np.unique(segmentation_data)

    segment = {'data': segmentation_data, 'unique_label': unique_label}

    outfile = open(os.path.join(atlas_folder, 'segment_pre_made.pkl'), 'wb')
    pickle.dump(segment, outfile)
    outfile.close()

    # pre-process atlas
    atlas_data, atlas_success = check_data_path_and_load(atlas_path)
    if not atlas_success:
        msg = 'Failed to load atlas volume data.'
        return msg

    new_atlas_data = atlas_data.copy()
    if mask_data is not None:
        # make atlas with mask
        new_atlas_data = new_atlas_data - np.min(new_atlas_data)
        for i in range(len(mask_data)):
            new_atlas_data[i][mask_data[i] == 0] = 0
        new_atlas_data = new_atlas_data / np.max(new_atlas_data)
    else:
        new_atlas_data = new_atlas_data - np.min(new_atlas_data)
        new_atlas_data = new_atlas_data / np.max(new_atlas_data)

    atlas_info = [
        {'name': 'anterior', 'values': np.arange(new_atlas_data.shape[0]) * voxel_size, 'units': 'um'},
        {'name': 'dorsal', 'values': np.arange(new_atlas_data.shape[1]) * voxel_size, 'units': 'um'},
        {'name': 'right', 'values': np.arange(new_atlas_data.shape[2]) * voxel_size, 'units': 'um'},
        {'vxsize': voxel_size,
         'Bregma': [bregma_coordinates[0], bregma_coordinates[1], bregma_coordinates[2]],
         'Lambda': [lambda_coordinates[0], lambda_coordinates[1], lambda_coordinates[2]]}
    ]

    atlas = {'data': new_atlas_data, 'info': atlas_info}

    atlas_data = atlas['data']
    atlas_info = atlas['info']

    outfile = open(os.path.join(atlas_folder, 'atlas_pre_made.pkl'), 'wb')
    pickle.dump(atlas, outfile)
    outfile.close()

    boundary = make_atlas_label_contour(atlas_folder, segmentation_data)

    msg = 'Atlas loaded successfully.'

    return atlas_data, atlas_info, segmentation_data, unique_label, boundary, msg






class AtlasMeshProcessor(object):
    def __init__(self, atlas_folder, atlas_data, segmentation_data, factor, level):
        meshdata = render_volume(atlas_data, atlas_folder, factor=factor, level=level)

        small_meshdata_list = render_small_volume(atlas_data, segmentation_data, atlas_folder,
                                                  factor=factor, level=level)


class AtlasLoader(object):
    def __init__(self, atlas_folder):
        pre_made_atlas_path = os.path.join(atlas_folder, 'atlas_pre_made.pkl')
        pre_made_segment_path = os.path.join(atlas_folder, 'segment_pre_made.pkl')
        pre_made_boundary_path = os.path.join(atlas_folder, 'contour_pre_made.pkl')

        pre_s_boundary_path = os.path.join(atlas_folder, 'sagital_contour_pre_made.pkl')
        pre_c_boundary_path = os.path.join(atlas_folder, 'coronal_contour_pre_made.pkl')
        pre_h_boundary_path = os.path.join(atlas_folder, 'horizontal_contour_pre_made.pkl')

        pre_made_label_info_path = os.path.join(atlas_folder, 'atlas_labels.pkl')

        chk1 = os.path.exists(pre_made_label_info_path)
        chk2 = os.path.exists(pre_made_atlas_path)
        chk3 = os.path.exists(pre_made_segment_path)

        if not np.all([chk1, chk2, chk3]):
            self.msg = 'Please pre-process the raw data of your desire atlas.'
            self.success = False
        else:
            # load label info
            try:
                infile = open(pre_made_label_info_path, 'rb')
                self.label_info = pickle.load(infile)
                infile.close()
                self.success = True
            except ValueError:
                self.msg = 'Please give the label information file. If you do not have one, ' \
                           'the maintainers are more than happy to help you to process one.'
                self.success = False
            # load atlas
            try:
                infile_atlas = open(pre_made_atlas_path, 'rb')
                atlas = pickle.load(infile_atlas)
                infile_atlas.close()

                self.atlas_data = atlas['data']
                self.atlas_info = atlas['info']
                # load segmentation
                infile_seg = open(pre_made_segment_path, 'rb')
                segment = pickle.load(infile_seg)
                infile_seg.close()

                self.segmentation_data = segment['data']

                data_shape = self.segmentation_data.shape
                print(data_shape)

                self.unique_label = segment['unique_label']
                self.success = True
            except (ValueError, pickle.UnpicklingError):
                self.msg = 'Please re-process atlas and label segmentation file.'
                self.success = False

        bnd_chk0 = os.path.exists(pre_made_boundary_path)
        bnd_chk1 = os.path.exists(pre_s_boundary_path)
        bnd_chk2 = os.path.exists(pre_c_boundary_path)
        bnd_chk3 = os.path.exists(pre_h_boundary_path)

        if not bnd_chk0:
            if not np.all([bnd_chk1, bnd_chk2, bnd_chk3]):
                self.msg = 'Please pre-process boundary file.'
                self.success = False
            else:
                # load boundary
                self.boundary = {}
                try:
                    infile = open(pre_s_boundary_path, 'rb')
                    sagital_contour_img = pickle.load(infile)
                    infile.close()

                    self.boundary['s_contour'] = sagital_contour_img
                    self.success = True
                except (ValueError, pickle.UnpicklingError, BrokenPipeError):
                    self.msg = 'Please re-process boundary file.'
                    self.success = False

                try:
                    infile = open(pre_c_boundary_path, 'rb')
                    coronal_contour_img = pickle.load(infile)
                    infile.close()

                    self.boundary['c_contour'] = coronal_contour_img
                    self.success = True
                except (ValueError, pickle.UnpicklingError, BrokenPipeError):
                    self.msg = 'Please re-process boundary file.'
                    self.success = False

                try:
                    infile = open(pre_h_boundary_path, 'rb')
                    horizontal_contour_img = pickle.load(infile)
                    infile.close()

                    self.boundary['h_contour'] = horizontal_contour_img
                    self.success = True
                except (ValueError, pickle.UnpicklingError, BrokenPipeError):
                    self.msg = 'Please re-process boundary file.'
                    self.success = False
        else:
            try:
                infile = open(pre_made_boundary_path, 'rb')
                bnd = pickle.load(infile)
                infile.close()

                self.boundary = bnd['data']
                self.success = True
            except (ValueError, pickle.UnpicklingError, BrokenPipeError):
                self.msg = 'Please re-process boundary file.'
                self.success = False











class AtlasMeshLoader(object):
    def __init__(self, atlas_folder):
        pre_made_meshdata_path = os.path.join(atlas_folder, 'atlas_meshdata.pkl')

