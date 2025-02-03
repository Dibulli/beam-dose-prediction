import os
import shutil
import numpy as np
from scipy.ndimage import zoom
import csv
from omegaconf import ListConfig
import random
from scipy.interpolate import RegularGridInterpolator


def clean_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(folder)

    '''
    # old version 
    if os.path.exists(self.check_img_path):
        shutil.rmtree(self.check_img_path)
    os.mkdir(self.check_img_path)
    '''


def shape_check_and_rescale(voxel, des_dim_y, des_dim_x):
    if np.ndim(voxel) == 3:
        vox_dim_y, vox_dim_x, vox_dim_z = voxel.shape
        if des_dim_y != vox_dim_y or des_dim_x != vox_dim_x:
            vox_data_type = voxel.dtype
            zoom_par = [des_dim_y / vox_dim_y, des_dim_x / vox_dim_x, 1]
            new_voxel = zoom(voxel, zoom_par, output=vox_data_type, order=1)
            return new_voxel
        else:
            return voxel

    if np.ndim(voxel) == 4:
        vox_dim_y, vox_dim_x, vox_dim_z, roi_dim = voxel.shape
        if des_dim_y != vox_dim_y or des_dim_x != vox_dim_x:
            vox_data_type = voxel.dtype
            zoom_par = [des_dim_y / vox_dim_y, des_dim_x / vox_dim_x, 1, 1]
            new_voxel = zoom(voxel, zoom_par, output=vox_data_type, order=1)
            return new_voxel
        else:
            return voxel


def read_recoder_list_csv(csv_file_name):
    with open(csv_file_name) as csvfile:
        recoder_list = []
        csvreader = csv.reader(csvfile)
        total_sample_num = 0
        patient_num = 0
        for row in csvreader:
            recoder_list.append(row)
            sample_num = int(row[2])
            total_sample_num += sample_num
            patient_num += 1
    return recoder_list, total_sample_num, patient_num


def get_all_files_in_the_folder(folder_i):
    file_list = []
    # get all files' path from directory
    for root, subdirs, filenames in os.walk(folder_i):
        file_list += map(lambda f: os.path.join(root, f), filenames)
    return file_list


def to_abs_path(path):
    if isinstance(path, ListConfig):
        path_list = []
        for path_i in path:
            if not os.path.isabs(path_i):
                abs_path_i = os.path.join(os.environ['ROOT_DIR'], path_i)
            else:
                abs_path_i = path_i
            path_list.append(abs_path_i)
        return path_list
    else:
        if not os.path.isabs(path):
            abs_path = os.path.join(os.environ['ROOT_DIR'], path)
        else:
            abs_path = path
        return abs_path


def random_path(cfg):
    train_path = to_abs_path(cfg.train_path)
    validation_path = to_abs_path(cfg.validation_path)
    rand_value = random.random() * 100
    if not cfg.val_flag:
        return train_path
    if rand_value < cfg.val_percentage:
        return validation_path
    else:
        return train_path


def dataset_split(cfg, total_sample_num):
    total_index = list(range(total_sample_num))
    val_index = random.sample(total_index, k=round(cfg.val_percentage * total_sample_num / 100))
    tr_index = list(set(total_index) - set(val_index))
    return tr_index, val_index


def dataset_cv_split(cfg, total_sample_num):
    total_index = list(range(total_sample_num))
    random.shuffle(total_index)

    fold_num = round(100 / cfg.val_percentage)
    cv_index_dict = {}

    for fold_i in range(fold_num):
        start_ind = round(fold_i * cfg.val_percentage / 100 * total_sample_num)
        end_ind = round((fold_i + 1) * cfg.val_percentage / 100 * total_sample_num)
        for i in range(start_ind, end_ind):
            cv_index_dict[total_index[i]] = fold_i

    return cv_index_dict


def cal_roi_balance_table(slice_mask):
    """
    main ideal:
    50% positive sample
    50% negative sample
    """

    z = slice_mask.shape[2]
    slice_mask_channel = slice_mask.shape[3]

    num_rois = np.zeros((z, slice_mask_channel))

    for channle_i in range(slice_mask_channel):
        num_rois[:, channle_i] = np.any(slice_mask[:, :, :, channle_i], axis=(0, 1))

    pos_num_rois = np.sum(num_rois, axis=0)
    neg_num_rois = z - pos_num_rois

    # deal with roi without positive and negative sample
    pos_num_rois[neg_num_rois == 0] = z / 2
    neg_num_rois[pos_num_rois == 0] = z / 2

    # avoid divided by 0
    pos_num_rois[pos_num_rois == 0] = 1e6
    neg_num_rois[neg_num_rois == 0] = 1e6

    p_rois = num_rois * (0.5 / pos_num_rois) + (1 - num_rois) * (0.5 / neg_num_rois)

    # p_roi = np.max(p_rois, axis=1) / sum(np.max(p_rois, axis=1))
    p_roi = np.sum(p_rois, axis=1) / sum(np.sum(p_rois, axis=1))

    return p_roi


def create_3d_data(slice_img, slice_mask):
    x = np.linspace(0, slice_img.shape[0] - 1, 96)
    y = np.linspace(0, slice_img.shape[1] - 1, 96)
    z = np.linspace(0, slice_img.shape[2] - 1, 96)

    img_interpolating_function = RegularGridInterpolator(
        (np.arange(slice_img.shape[0]), np.arange(slice_img.shape[1]), np.arange(slice_img.shape[2])), slice_img)

    pii = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    img_3d_rough = img_interpolating_function(pii).reshape(96, 96, 96)

    img_3d_fine = np.zeros((96, 96, 96, slice_mask.shape[3]), dtype=np.float)
    mask_3d_rough = np.zeros((96, 96, 96, slice_mask.shape[3]), dtype=np.int)
    mask_3d_fine = np.zeros((96, 96, 96, slice_mask.shape[3]), dtype=np.int)

    for mask_i in range(mask_3d_rough.shape[3]):
        mask_interpolating_function = RegularGridInterpolator(
            (np.arange(slice_img.shape[0]), np.arange(slice_img.shape[1]), np.arange(slice_img.shape[2])),
            slice_mask[:, :, :, mask_i])
        mask_3d_rough[:, :, :, mask_i] = np.round(mask_interpolating_function(pii).reshape(96, 96, 96))

        x_start = max(np.min(np.where(np.any(slice_mask[:, :, :, mask_i], axis=(1, 2)))) - 10, 0)
        x_end = min(np.max(np.where(np.any(slice_mask[:, :, :, mask_i], axis=(1, 2)))) + 10, slice_img.shape[0] - 1)
        y_start = max(np.min(np.where(np.any(slice_mask[:, :, :, mask_i], axis=(0, 2)))) - 10, 0)
        y_end = min(np.max(np.where(np.any(slice_mask[:, :, :, mask_i], axis=(0, 2)))) + 10, slice_img.shape[1] - 1)
        z_start = max(np.min(np.where(np.any(slice_mask[:, :, :, mask_i], axis=(0, 1)))) - 2, 0)
        z_end = min(np.max(np.where(np.any(slice_mask[:, :, :, mask_i], axis=(0, 1)))) + 2, slice_img.shape[2] - 1)

        x_fine = np.linspace(x_start, x_end, 96)
        y_fine = np.linspace(y_start, y_end, 96)
        z_fine = np.linspace(z_start, z_end, 96)

        pii_fine = np.array(np.meshgrid(x_fine, y_fine, z_fine)).T.reshape(-1, 3)
        img_3d_fine[:, :, :, mask_i] = img_interpolating_function(pii_fine).reshape(96, 96, 96)
        mask_3d_fine[:, :, :, mask_i] = np.round(mask_interpolating_function(pii_fine).reshape(96, 96, 96))

    return img_3d_rough, img_3d_fine, mask_3d_rough, mask_3d_fine
