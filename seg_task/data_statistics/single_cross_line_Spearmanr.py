import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from pydicom.filereader import dcmread
import cv2
import torch
import objgraph
from seg_task.Beams_Contour_Generator import BeamContourGeneration
# from seg_task.segformer.seg_model_segformer import SegModel
import sys
import time
from lib.utilities import shape_check_and_rescale
# pred = np.load("/Users/mr.chai/Desktop/02209225.npy")
import torch.nn as nn
import torch
from torchvision.transforms.functional import rotate
import pandas as pd
import csv
import einops
import os
from sklearn.metrics import mean_absolute_error as MAE
from scipy import ndimage
from scipy import stats

class h5_file_rotation():
    def __init__(self, input_h5_data, pred_dose_path, min_z, max_z):
        self.img = input_h5_data["slice_img"]
        self.mask = input_h5_data["slice_mask"]
        self.beam = input_h5_data["slice_beam"]
        self.dose = input_h5_data["slice_dose"]
        self.dim_x = self.img.shape[0]
        self.dim_y = self.img.shape[1]
        self.dim_z = self.img.shape[2]
        self.pred_raw = np.load(pred_dose_path)
        self.cut_pred = self.pred_raw[:, :, min_z:max_z, :]
        self.cut_img = self.img[:, :, min_z:max_z]
        dose_shift_list = [4, 5, 6, 7, 8, 0, 1, 2, 3, -1]
        self.shifted_dose = np.array([self.dose[:, :, :, i] for i in dose_shift_list])
        self.real_dose = einops.rearrange(self.shifted_dose, 'b h w z -> h w z b')
        self.ptv = np.sum(self.beam[1:, :, :, min_z:max_z, :], axis=0, keepdims=False)
        self.pdd = self.beam[0, :, :, min_z:max_z, :]
    # def rotation_center(self):
    #     ptv60 = self.mask[:, :, :, 2]
    #     x_all, y_all = np.nonzero(ptv60)[0], np.nonzero(ptv60)[1]
    #     cent_x, cent_y = int(x_all.mean().round()), int(y_all.mean().round())
    #     return cent_x, cent_y

    def line_and_pdd_extraction_2d(self, single_dose_slice, pred_single_slice, input_pdd_slice, beam_id, input_ct_slice, input_ptvs_slice):
        beam_angle = beam_id * 40
        xi, yi = np.meshgrid(np.arange(self.dim_x), np.arange(self.dim_x))
        max_x = np.unravel_index(single_dose_slice.argmax(), single_dose_slice.shape)[1]
        max_y = np.unravel_index(single_dose_slice.argmax(), single_dose_slice.shape)[0]
        if beam_id == 0:
            valid_real_dose = single_dose_slice[:, max_x]
            valid_pred_dose = pred_single_slice[:, max_x]
            valid_pdd_slice = input_pdd_slice[:, max_x]
            valid_ct_slice = input_ct_slice[:, max_x]
            valid_ptvs_slice = input_ptvs_slice[:, max_x]
        else:
            ind1 = np.where(np.round((yi - max_y) + (xi - max_x) * np.tan((90 - beam_angle) * np.pi / 180)) == 0)
            valid_real_dose = single_dose_slice[ind1]
            valid_pred_dose = pred_single_slice[ind1]
            valid_pdd_slice = input_pdd_slice[ind1]
            valid_ct_slice = input_ct_slice[ind1]
            valid_ptvs_slice = input_ptvs_slice[ind1]

        ind_body = np.where(valid_real_dose != 0)
        real_dose = valid_real_dose[ind_body]
        pred_dose = valid_pred_dose[ind_body]
        pdd_slice = valid_pdd_slice[ind_body]
        ct_slice = valid_ct_slice[ind_body]
        ptv_slice = valid_ptvs_slice[ind_body]

        return real_dose, pred_dose, pdd_slice, ct_slice, ptv_slice

    def cross_and_ptv_extraction_2d(self, single_dose_slice, pred_single_slice, input_ptv_slice,  beam_id, input_ct_slice, input_pdd_slice):
        beam_angle = beam_id * 40
        xi, yi = np.meshgrid(np.arange(self.dim_x), np.arange(self.dim_x))
        max_x = np.unravel_index(single_dose_slice.argmax(), single_dose_slice.shape)[1]
        max_y = np.unravel_index(single_dose_slice.argmax(), single_dose_slice.shape)[0]
        if beam_id == 0:
            valid_real_dose = single_dose_slice[max_y, :]
            valid_pred_dose = pred_single_slice[max_y, :]
            valid_ptv_slice = input_ptv_slice[max_y, :]
            valid_pdd_slice = input_pdd_slice[max_y, :]
            valid_ct_slice = input_ct_slice[max_y, :]
        else:
            ind1 = np.where(np.round((yi - max_y) - (xi - max_x) / (np.tan((90 - beam_angle) * np.pi / 180))) == 0)
            valid_real_dose = single_dose_slice[ind1]
            valid_pred_dose = pred_single_slice[ind1]
            valid_ptv_slice = input_ptv_slice[ind1]
            valid_pdd_slice = input_pdd_slice[ind1]
            valid_ct_slice = input_ct_slice[ind1]

        ind_body = np.where(valid_real_dose != 0)
        real_dose = valid_real_dose[ind_body]
        pred_dose = valid_pred_dose[ind_body]
        ptv_slice = valid_ptv_slice[ind_body]
        ct_slice = valid_ct_slice[ind_body]
        pdd_slice = valid_pdd_slice[ind_body]

        return real_dose, pred_dose, pdd_slice, ct_slice, ptv_slice


if __name__ == "__main__":
    df = pd.read_csv("/Volumes/NPC预测数据盘/单野预测2024/batch_test_h5/data_recoder.csv", header=None)
    h5_name_list = df.iloc[:, 0]
    id_list = df.iloc[:, 1]
    min_z_list = df.iloc[:, 3]
    max_z_list = df.iloc[:, 4]
    dim_z_list = df.iloc[:, 2]
    """待完成：批量预测单野剂量分布（20 cases）。"""
    for beam_id in range(9):
        joint_id = beam_id * 40
        # with open('/Volumes/NPC预测数据盘/单野预测2024/line_spearmann/' + str(joint_id) +  ".csv", 'w+') as csv_file:
        with open('/Volumes/NPC预测数据盘/单野预测2024/cross_spearmann/' + str(joint_id) + ".csv", 'w+') as csv_file:

            csv_file.write(', z, real_ct, real_pdd, real_ptv, pred_ct, pred_pdd, pred_ptv')
            csv_file.write('\n, ')

        for pt in range(len(h5_name_list)):
            h5_file = h5.File("/Volumes/NPC预测数据盘/单野预测2024/batch_test_h5/" + h5_name_list[pt] +  ".h5", 'r')
            pred_path = "/Volumes/NPC预测数据盘/单野预测2024/batch_pred_npy/" + str(id_list[pt]) + "_-1.npy"
            min_z = min_z_list[pt]
            max_z = max_z_list[pt]
            dim_z = dim_z_list[pt]
            rot = h5_file_rotation(h5_file, pred_path, min_z, max_z)

            for z in range(dim_z - 1):
                real = rot.real_dose[:, :, z, beam_id]
                pred = rot.cut_pred[:, :, z, beam_id]
                pdd = rot.pdd[:, :, z, beam_id]
                ptv = rot.ptv[:, :, z, beam_id]
                ct_slice = rot.cut_img[:, :, z]
                # real_line, pred_line, pdd_line, ct_line, ptv_line = rot.line_and_pdd_extraction_2d(real, pred, pdd,
                #                                                                                    beam_id, ct_slice,
                #                                                                                    ptv)
                real_line, pred_line, pdd_line, ct_line, ptv_line = rot.cross_and_ptv_extraction_2d(real, pred, ptv,
                                                                                                   beam_id, ct_slice,
                                                                                                   pdd)
                # if len(real_line) < 20 :
                #     continue
                real_pdd_line_corr = stats.spearmanr(real_line, pdd_line)[0]
                pred_pdd_line_corr = stats.spearmanr(pred_line, pdd_line)[0]
                real_ct_line_corr = stats.spearmanr(real_line, ct_line)[0]
                pred_ct_line_corr = stats.spearmanr(pred_line, ct_line)[0]
                real_ptv_corr = stats.spearmanr(real_line, ptv_line)[0]
                pred_ptv_corr = stats.spearmanr(pred_line, ptv_line)[0]

                # print(z)

                # print('\n')

                # with open('/Volumes/NPC预测数据盘/单野预测2024/line_spearmann/' + str(joint_id) +  ".csv", 'a+') as csv_file:
                with open('/Volumes/NPC预测数据盘/单野预测2024/cross_spearmann/' + str(joint_id) + ".csv", 'a+') as csv_file:

                    csv_file.write(
                        ", {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(real_ct_line_corr, real_pdd_line_corr,
                                                                                  real_ptv_corr,
                                                                                  pred_ct_line_corr, pred_pdd_line_corr,
                                                                                  pred_ptv_corr))
                    csv_file.write('\n, ')
            # print("One pt create over!")
        print("One beam create over!")
    # h5_file = h5.File("/Volumes/NPC预测数据盘/临时存储/test_h5_256/e9278201.h5", 'r')
    # pred_path = "/Volumes/NPC预测数据盘/单野预测2024/临时存储2/single + masked/2100943743_-1.npy"



# with open('/Volumes/NPC预测数据盘/单野预测2024/beam0/' + "cross_and_line_corr4.csv", 'w+') as csv_file:
#     csv_file.write(', z, max_x, max_y, real_line, pred_line, real_cross, pred_cross')
#     csv_file.write('\n, ')
# # f = h5.File("/Volumes/NPC预测数据盘/临时存储/3e158adf.h5", 'r')
# f = h5.File("/Volumes/NPC预测数据盘/临时存储/test_h5_256/e9278201.h5", 'r')
# # beam_id = 8
# # mask_id = 1
# # i = f["slice_img"]
# m = f["slice_mask"]
# body = m[:, :, :, 0]
# b = f["slice_beam"]
# # oars = np.sum(m[:, :, 30, 4:], axis=-1, keepdims=False)
# # k = np.abs(b[mask_id, :, :, 30, beam_id])
# # # plt.imshow(k, vmin=0, vmax=0.3, cmap='jet')
# # plt.imshow(i[:, :, 30], cmap='Greys', vmin=0, vmax=0.8)
# # # plt.contour(m[:, :, 30, 0], linewidths=1, colors='black')
# # # plt.contour(m[:, :, 30, mask_id], linewidths=1, colors='darkorange')
# # # plt.contour(oars, colors='lightgreen')
# #
# # plt.show()
# pred_dose = np.load("/Volumes/NPC预测数据盘/单野预测2024/临时存储2/single + masked/2100943743_-1.npy")
# # pred_dose = np.load("/Volumes/NPC预测数据盘/AutoPlan_2024/AutoPlan_Predicted_dose/unet/2100943743_-1.npy")
# cut_pred = pred_dose[:, :, 43:129, :]
#
# real_dose = f["slice_dose"]
# dose_shift_list = [4, 5, 6, 7, 8, 0, 1, 2, 3, -1]
# shifted_dose = np.array([real_dose[:, :, :, i] for i in dose_shift_list])
# dose = einops.rearrange(shifted_dose, 'b h w z -> h w z b')
#
# z_dim = b.shape[3]
# for z in range(z_dim):
# # z=0
#     body_slice = body[:, :, z]
#
#     pdd = b[0, :, :, z, 0]
#     # beam_slice = b[0, :, :, 30, 0]
#     ptv = np.sum(b[1:, :, :, z, 0], axis=0, keepdims=False)
#     slice_dose = dose[:, :, z, 0]
#     max_x = np.unravel_index(slice_dose.argmax(), slice_dose.shape)[0]
#     max_y = np.unravel_index(slice_dose.argmax(), slice_dose.shape)[1]
#     # max_x = 118
#     # max_y = 118
#     body_line = body_slice[:, max_y]
#     body_cross = body_slice[max_x, :]
#     sample_points_line = np.nonzero(body_line)
#     sample_points_cross = np.nonzero(body_cross)
#     if len(sample_points_line[0]) < 40 or len(sample_points_cross[0]) < 40:
#         continue
#
#     real_line = slice_dose[:, max_y]
#     pred_line = cut_pred[:, max_y, z, 0]
#     beam_line = pdd[:, max_y]
#
#     real_cross = slice_dose[max_x, :]
#     pred_cross = cut_pred[max_x, :, z, 0]
#     ptv_cross = ptv[max_x, :]
#
#
#     #
#     real_line_corr = stats.spearmanr(real_line[sample_points_line], beam_line[sample_points_line])[0]
#     pred_line_corr = stats.spearmanr(pred_line[sample_points_line] , beam_line[sample_points_line])[0]
#     real_cross_corr = stats.spearmanr(real_cross[sample_points_cross], ptv_cross[sample_points_cross])[0]
#     pred_cross_corr = stats.spearmanr(pred_cross[sample_points_cross] , ptv_cross[sample_points_cross])[0]
#     with open('/Volumes/NPC预测数据盘/单野预测2024/beam0/' + "cross_and_line_corr4.csv", 'a+') as csv_file:
#         csv_file.write("{:d}, {:d}, {:d}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(z, max_x, max_y, real_line_corr, pred_line_corr,
#                                                                            real_cross_corr, pred_cross_corr))
#         csv_file.write('\n, ')
#
# # print(real_line_corr)
# # print(pred_line_corr)
# # print(real_cross_corr)
# # print(pred_cross_corr)
# # plt.plot(real_cross)
# # plt.plot(ptv_cross)
# # plt.plot(pred_cross)
#
# # plt.plot(beam_line)
# # plt.plot(real_line)
# # plt.plot(pred_line)
# # # # plt.plot(real_line / np.max(real_line))
# # # # plt.plot(real_cross)
# # # # # plt.plot(pred_line / np.max(pred_line))
# # # # # plt.plot(beam_line)
# # plt.show()





