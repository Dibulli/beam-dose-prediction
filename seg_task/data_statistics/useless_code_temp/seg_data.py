import logging
import os
import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from lib.color_palette import *
from lib.utilities import clean_folder
from lib.utilities import read_recoder_list_csv
from lib.utilities import to_abs_path

logger = logging.getLogger(__name__)


class SegData(IterableDataset):
    def __init__(self, cfg, data_type='train'):
        IterableDataset.__init__(self)

        self.cfg = cfg

        # parameter for data
        self.data_type = data_type

        if self.data_type == 'train':
            self.data_path = to_abs_path(cfg.train_path)
            self.augmentation = True
        elif self.data_type == 'validation':
            self.data_path = to_abs_path(cfg.validation_path)
            self.augmentation = False
            self.lr_flip = False
        else:
            self.data_path = to_abs_path(cfg.test_path)
            self.augmentation = False
            self.lr_flip = False

        # self.body_channel = 0
        # for roi in self.cfg.roi_dicts:
        #     if roi.name.lower() == 'body':
        #         self.body_channel = roi.mask_channel

        # parameter for data
        self.recoder_list = []
        self.sample_num = 0
        self.patient_num = 0

        # read configure and recoder list
        self.read_recoder_list()

        logger.info('iterable data initialize successful')
        logger.info('data type: ' + self.data_type)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # bug? in linux each time the seed is same
        np.random.seed(None)

        if worker_info is None:
            recoder_list = self.recoder_list
        else:
            sub_list_id = [i for i in range(worker_info.id, self.patient_num, worker_info.num_workers)]
            recoder_list = [self.recoder_list[i] for i in sub_list_id]

        random.shuffle(recoder_list)
        for record_idx, record_item in enumerate(recoder_list):

            data_id = record_item[0]
            pt_id = record_item[1]
            hdf5_filename = self.data_path + os.sep + data_id + '.h5'
            logger.debug('reading ' + hdf5_filename)
            logger.debug('this data come from ' + pt_id)
            hdf5_file = h5py.File(hdf5_filename, 'r')
            slice_img = hdf5_file["slice_img"][:, :, :]
            slice_mask = hdf5_file["slice_mask"][:, :, :]
            slice_dose = hdf5_file["slice_dose"][:, :, :]
            # p_roi = hdf5_file["p_roi"][:]
            hdf5_file.close()
            # print("slice_img", slice_img.shape)
            # print("slice_mask", slice_mask.shape)
            # print("slice_dose", slice_dose.shape)

            z_dim = slice_img.shape[2]
            logger.debug('This patient have ' + str(z_dim) + ' samples.')

            half_dim = int((self.cfg.channels_of_input_images - 1) / 2)
            # if self.data_type == 'train':
            #     if self.cfg.equal_pt_weight:
            #         z_list = np.random.choice(z_dim, size=self.cfg.sample_per_pt, replace=True, p=p_roi)
            #     else:
            #         z_list = np.random.choice(z_dim, size=z_dim, replace=True, p=p_roi)
            # else:
                # use all data
            z_list = [i for i in range(z_dim)]
                # not use begin and end
                # z_list = [i for i in range(half_dim, z_dim - half_dim)]

            for z in z_list:
            # z = 44
            # np_img = np.zeros(shape=(self.cfg.dim_y, self.cfg.dim_x, self.cfg.channels_of_input_images), dtype=np.float32)
                np_img = np.zeros(shape=(self.cfg.dim_y, self.cfg.dim_x, 1), dtype=np.float32)

                np_dose = slice_dose[:, :, z].astype(np.float32)
                np_dose = np_dose.reshape((self.cfg.dim_y, self.cfg.dim_x, 1))
                # print("np_dose.shape:", np_dose.shape)
                np_mask = slice_mask[:, :, z, :].astype(np.float32)

                # avoid out the matrix
                # for rel_slice_loc in range(self.cfg.channels_of_input_images):
                for rel_slice_loc in range(1):

                    z_loc = max(z + rel_slice_loc - half_dim, 0)
                    z_loc = min(z_loc, z_dim - 1)
                    np_img[:, :, rel_slice_loc] = slice_img[:, :, z_loc]

                np_img_list = [np_img]
                np_dose_list = [np_dose]
                # augmentation add here
                # if self.augmentation:
                #     np_img_list, np_dose_list, np_mask = self.run_augmentation(np_img_list, np_dose_list, np_mask)
                # np_img_list, np_dose_list, np_mask = self.run_augmentation(np_img_list, np_dose_list,
                #                                                                            np_mask)
                np_img = np.concatenate(np_img_list, axis=2)
                np_img = np.array(np_img, dtype=np.float32)
                np_img = np.rollaxis(np_img, 2, 0)

                np_dose = np.concatenate(np_dose_list, axis=2)
                np_dose = np.array(np_dose, dtype=np.float32)
                np_dose = np.rollaxis(np_dose, 2, 0)

                out_mask = np.zeros(shape=(self.cfg.channels_of_input_masks, self.cfg.dim_y, self.cfg.dim_x),
                                    dtype=np.float32)

                for roi_ind_in_conf, single_roi_conf in enumerate(self.cfg.roi_dicts):
                    out_mask[single_roi_conf.mask_channel, :, :] = np_mask[:, :, roi_ind_in_conf]

                # box_dose = np_dose * out_mask[self.body_channel, :, :]

                mask_with_img = np.concatenate((np_img, out_mask), axis=0)

                # print("np_img:", np_img.shape)
                # print("mask with img:", mask_with_img.shape)
                # print("out_mask:", out_mask.shape)
                # print("np_dose:", np_dose.shape)
                np.save("/Users/mr.chai/Desktop/mask_with_img.npy", mask_with_img)
                yield np_img, out_mask, np_dose, pt_id, z, mask_with_img

    def run_augmentation(self, img_array_list, dose_array_list, mask_array):
        dim_x = self.cfg.dim_x
        dim_y = self.cfg.dim_y

        # shape disturbance
        # pts1 = np.float32([[0, 0], [dim_x, 0], [dim_x, dim_y]])
        # p0_x_new = 0 + dim_x * (np.random.rand() / 2.5 - 0.2)
        # p0_y_new = 0 + dim_y * (np.random.rand() / 2.5 - 0.2)
        # p1_x_new = dim_x + dim_x * (np.random.rand() / 2.5 - 0.2)
        # p1_y_new = 0 + dim_y * (np.random.rand() / 2.5 - 0.2)
        # p2_x_new = dim_x + dim_x * (np.random.rand() / 2.5 - 0.2)
        # p2_y_new = dim_y + dim_y * (np.random.rand() / 2.5 - 0.2)
        # pts2 = np.float32([[p0_x_new, p0_y_new], [p1_x_new, p1_y_new], [p2_x_new, p2_y_new]])
        # affine_mat = cv2.getAffineTransform(pts1, pts2)

        perturbed_img_array_list = []
        perturbed_dose_array_list = []

        for array_i, array_d_i in zip(img_array_list, dose_array_list):
            # disturbance_plus = np.random.rand() / 2.5 - 0.2
            disturbance_mul = np.random.rand() / 5 + 0.9
            array_i = array_i * disturbance_mul
            # array_i = array_i + np.ones(array_i.shape) * disturbance_plus
            # train_img_min_value = np.amin(array_i)
            # train_dose_min_value = np.amin(array_d_i)
            #
            # array_i = cv2.warpAffine(src=array_i, M=affine_mat, dsize=(dim_x, dim_y),
            #                          dst=np.ones(array_i.shape) * train_img_min_value,
            #                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            #
            # array_d_i = cv2.warpAffine(src=array_d_i, M=affine_mat, dsize=(dim_x, dim_y),
            #                            dst=np.ones(array_d_i.shape) * train_dose_min_value,
            #                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # gaussian noise
            # nosie_sigma = np.random.rand() / 10
            # array_i = array_i + np.random.normal(0, nosie_sigma, array_i.shape)

            # smooth or sharp
            kernel_center = 2 ** (np.random.rand() - 0.5)
            kernel_around = (1 - kernel_center) / 8
            kernel = kernel_around * np.ones((3, 3), dtype=float)
            kernel[1, 1] = kernel_center

            array_i = cv2.filter2D(array_i, -1, kernel)
            # array_d_i = cv2.filter2D(array_d_i, -1, kernel)

            # array_i = np.clip(array_i, 0, 1)
            # array_d_i = np.clip(array_d_i, 0, 1)
            array_i = array_i.reshape((dim_x, dim_y, 1))
            array_d_i = array_d_i.reshape((dim_x, dim_y, 1))


        # mask_array = cv2.warpAffine(src=mask_array, M=affine_mat, dsize=(dim_x, dim_y),
        #                             dst=np.zeros_like(mask_array),
        #                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        mask_array = np.round(mask_array)
        mask_array.astype(np.float32)

        if random.random() < 0.5:
            array_i = np.fliplr(array_i)
            array_d_i = np.fliplr(array_d_i)
            mask_array = np.fliplr(mask_array)

        perturbed_img_array_list.append(array_i)
        perturbed_dose_array_list.append(array_d_i)

        return perturbed_img_array_list, perturbed_dose_array_list, mask_array

    # def run_augmentation(self, img_array_list, mask_array):
    #     dim_x = self.cfg.dim_x
    #     dim_y = self.cfg.dim_y
    #     fliplr_rand = random.uniform(0, 1)
    #     flipud_rand = random.uniform(0, 1)
    #     # shape disturbance
    #     pts1 = np.float32([[0, 0], [dim_x, 0], [dim_x, dim_y]])
    #     p0_x_new = 0 + dim_x * (np.random.rand() / 2.5 - 0.2)
    #     p0_y_new = 0 + dim_y * (np.random.rand() / 2.5 - 0.2)
    #     p1_x_new = dim_x + dim_x * (np.random.rand() / 2.5 - 0.2)
    #     p1_y_new = 0 + dim_y * (np.random.rand() / 2.5 - 0.2)
    #     p2_x_new = dim_x + dim_x * (np.random.rand() / 2.5 - 0.2)
    #     p2_y_new = dim_y + dim_y * (np.random.rand() / 2.5 - 0.2)
    #     pts2 = np.float32([[p0_x_new, p0_y_new], [p1_x_new, p1_y_new], [p2_x_new, p2_y_new]])
    #     affine_mat = cv2.getAffineTransform(pts1, pts2)
    #
    #     perturbed_img_array_list = []
    #     if fliplr_rand > 1.1:
    #         if flipud_rand < 0.5:
    #             for array_i in img_array_list:
    #                 disturbance_plus = np.random.rand() / 2.5 - 0.2
    #                 disturbance_mul = np.random.rand() / 2.5 + 0.8
    #                 array_i = array_i * disturbance_mul
    #                 array_i = array_i + np.ones(array_i.shape) * disturbance_plus
    #                 train_img_min_value = np.amin(array_i)
    #
    #                 array_i = cv2.warpAffine(src=array_i, M=affine_mat, dsize=(dim_x, dim_y),
    #                                          dst=np.ones(array_i.shape) * train_img_min_value,
    #                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #                 array_i = np.fliplr(array_i)
    #                 array_i = np.flipud(array_i)
    #
    #                 # smooth or sharp
    #                 kernel_center = 10 ** (np.random.rand() - 0.5)
    #                 kernel_around = (1 - kernel_center) / 8
    #                 kernel = kernel_around * np.ones((3, 3), dtype=float)
    #                 kernel[1, 1] = kernel_center
    #
    #                 array_i = cv2.filter2D(array_i, -1, kernel)
    #
    #                 array_i = np.clip(array_i, 0, 1)
    #                 perturbed_img_array_list.append(array_i)
    #
    #             mask_array = cv2.warpAffine(src=mask_array, M=affine_mat, dsize=(dim_x, dim_y),
    #                                         dst=np.zeros_like(mask_array),
    #                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #             mask_array = np.round(mask_array)
    #             mask_array = np.fliplr(mask_array)
    #             mask_array = np.flipud(mask_array)
    #             mask_array.astype(np.float32)
    #         else:
    #             for array_i in img_array_list:
    #                 disturbance_plus = np.random.rand() / 2.5 - 0.2
    #                 disturbance_mul = np.random.rand() / 2.5 + 0.8
    #                 array_i = array_i * disturbance_mul
    #                 array_i = array_i + np.ones(array_i.shape) * disturbance_plus
    #                 train_img_min_value = np.amin(array_i)
    #
    #                 array_i = cv2.warpAffine(src=array_i, M=affine_mat, dsize=(dim_x, dim_y),
    #                                          dst=np.ones(array_i.shape) * train_img_min_value,
    #                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #                 array_i = np.fliplr(array_i)
    #                 # smooth or sharp
    #                 kernel_center = 10 ** (np.random.rand() - 0.5)
    #                 kernel_around = (1 - kernel_center) / 8
    #                 kernel = kernel_around * np.ones((3, 3), dtype=float)
    #                 kernel[1, 1] = kernel_center
    #
    #                 array_i = cv2.filter2D(array_i, -1, kernel)
    #
    #                 array_i = np.clip(array_i, 0, 1)
    #                 perturbed_img_array_list.append(array_i)
    #
    #             mask_array = cv2.warpAffine(src=mask_array, M=affine_mat, dsize=(dim_x, dim_y),
    #                                         dst=np.zeros_like(mask_array),
    #                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #             mask_array = np.round(mask_array)
    #             mask_array = np.fliplr(mask_array)
    #             mask_array.astype(np.float32)
    #     else:
    #         if flipud_rand < 0.5:
    #             for array_i in img_array_list:
    #                 disturbance_plus = np.random.rand() / 2.5 - 0.2
    #                 disturbance_mul = np.random.rand() / 2.5 + 0.8
    #                 array_i = array_i * disturbance_mul
    #                 array_i = array_i + np.ones(array_i.shape) * disturbance_plus
    #                 train_img_min_value = np.amin(array_i)
    #
    #                 array_i = cv2.warpAffine(src=array_i, M=affine_mat, dsize=(dim_x, dim_y),
    #                                          dst=np.ones(array_i.shape) * train_img_min_value,
    #                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #                 array_i = np.flipud(array_i)
    #                 # smooth or sharp
    #                 kernel_center = 10 ** (np.random.rand() - 0.5)
    #                 kernel_around = (1 - kernel_center) / 8
    #                 kernel = kernel_around * np.ones((3, 3), dtype=float)
    #                 kernel[1, 1] = kernel_center
    #
    #                 array_i = cv2.filter2D(array_i, -1, kernel)
    #
    #                 array_i = np.clip(array_i, 0, 1)
    #                 perturbed_img_array_list.append(array_i)
    #
    #             mask_array = cv2.warpAffine(src=mask_array, M=affine_mat, dsize=(dim_x, dim_y),
    #                                         dst=np.zeros_like(mask_array),
    #                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #             mask_array = np.round(mask_array)
    #             mask_array = np.flipud(mask_array)
    #             mask_array.astype(np.float32)
    #         else:
    #             for array_i in img_array_list:
    #                 disturbance_plus = np.random.rand() / 2.5 - 0.2
    #                 disturbance_mul = np.random.rand() / 2.5 + 0.8
    #                 array_i = array_i * disturbance_mul
    #                 array_i = array_i + np.ones(array_i.shape) * disturbance_plus
    #                 train_img_min_value = np.amin(array_i)
    #
    #                 array_i = cv2.warpAffine(src=array_i, M=affine_mat, dsize=(dim_x, dim_y),
    #                                          dst=np.ones(array_i.shape) * train_img_min_value,
    #                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #                 # smooth or sharp
    #                 kernel_center = 10 ** (np.random.rand() - 0.5)
    #                 kernel_around = (1 - kernel_center) / 8
    #                 kernel = kernel_around * np.ones((3, 3), dtype=float)
    #                 kernel[1, 1] = kernel_center
    #
    #                 array_i = cv2.filter2D(array_i, -1, kernel)
    #
    #                 array_i = np.clip(array_i, 0, 1)
    #                 perturbed_img_array_list.append(array_i)
    #
    #             mask_array = cv2.warpAffine(src=mask_array, M=affine_mat, dsize=(dim_x, dim_y),
    #                                         dst=np.zeros_like(mask_array),
    #                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #             mask_array = np.round(mask_array)
    #             mask_array.astype(np.float32)
    #     return perturbed_img_array_list, mask_array

    def read_recoder_list(self):
        recoder_file = self.data_path + os.sep + "data_recoder.csv"
        logger.info('read file: ' + recoder_file)
        if os.path.exists(recoder_file):
            self.recoder_list, self.sample_num, self.patient_num = read_recoder_list_csv(recoder_file)
            if self.cfg.equal_pt_weight:
                self.sample_num = self.patient_num * self.cfg.sample_per_pt
        else:
            logger.info('no file: ' + recoder_file)


def dataset_check(cfg, data_type):
    check_img_path = to_abs_path(cfg.check_img_path)
    clean_folder(check_img_path)

    dl_ds = SegData(cfg, data_type)

    roi_dicts = cfg.roi_dicts
    count_num = np.zeros((1, dl_ds.sample_num), dtype=int)

    for epoch_i in range(cfg.epochs):
        data_loader = DataLoader(dl_ds, batch_size=cfg.batch_size,
                                 pin_memory=True)

        for batch_i, batch_data in enumerate(data_loader):
            batch_img = batch_data[0].numpy()
            batch_mask = batch_data[1].numpy()
            batch_dose = batch_data[2].numpy()
            batch_pt_id = batch_data[3]
            batch_z = batch_data[4].numpy()

            for sample_i in range(len(batch_pt_id)):
                logger.info('epoch [{}] batch [{}] sample [{}]'.format(
                    str(epoch_i), str(batch_i), str(sample_i)))
                pt_id = batch_pt_id[sample_i]

                combine_check_img_rgb = None
                check_img = np.array(batch_img[sample_i, round(cfg.channels_of_check_images / 2), :, :] * 255, dtype=np.uint8)
                check_img_rgb = cv2.cvtColor(check_img, cv2.COLOR_GRAY2RGB)
                check_dose = np.array(batch_dose[sample_i, :, :].transpose(1, 2, 0) * 2.55).astype(np.uint8)
                # print(check_dose.shape)
                check_dose_rgb = cv2.applyColorMap(check_dose, cv2.COLORMAP_JET)

                for single_roi_conf in roi_dicts:
                    roi_color = set1[single_roi_conf.mask_channel]
                    roi_check_img = batch_mask[sample_i, single_roi_conf.mask_channel, :, :]

                    roi_check_img_rgb = np.repeat(roi_check_img[:, :, np.newaxis], 3, axis=2)
                    roi_check_img_rgb = roi_check_img_rgb * np.array(roi_color, dtype=int)
                    roi_check_img_rgb = np.array(roi_check_img_rgb, dtype=np.uint8)

                    if combine_check_img_rgb is None:
                        combine_check_img_rgb = cv2.addWeighted(check_img_rgb, 1.0, roi_check_img_rgb,
                                                                cfg.roi_color_overlay_alpha, 0)
                    else:
                        combine_check_img_rgb = cv2.addWeighted(combine_check_img_rgb, 1.0, roi_check_img_rgb,
                                                                cfg.roi_color_overlay_alpha, 0)

                    if cfg.gen_mask_img_for_each_roi:
                        file_name = '{0:s}_[{1:s}]_e[{2:s}]_b[{3:s}]_s[{4:s}]_roi_[{5:s}].png'.format(
                            data_type, pt_id, str(epoch_i), str(batch_i), str(sample_i), single_roi_conf.name)
                        file_path = check_img_path + os.sep + file_name
                        cv2.imwrite(file_path, roi_check_img_rgb)


                file_name = '{0:s}_[{1:s}]_e[{2:s}]_b[{3:s}]_s[{4:s}].png'.format(
                    data_type, pt_id, str(epoch_i), str(batch_i), str(sample_i))
                file_path = check_img_path + os.sep + file_name
                cv2.imwrite(file_path, combine_check_img_rgb)
                dose_file_name = '{0:s}_[{1:s}]_e[{2:s}]_b[{3:s}]_s[{4:s}]_dose.png'.format(
                    data_type, pt_id, str(epoch_i), str(batch_i), str(sample_i))
                dose_file_path = check_img_path + os.sep + dose_file_name
                cv2.imwrite(dose_file_path, check_dose_rgb)

                count_num[0, batch_z[sample_i]] = count_num[0, batch_z[sample_i]] + 1

        # for data check
        # logger.info('count:')
        # logger.info(count_num)
