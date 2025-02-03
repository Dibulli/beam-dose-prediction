import pydicom
import os
import h5py as h5
import matplotlib.pyplot as plt
import einops
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd
import torch
import math

import seaborn as sns

# df = pd.read_csv("/Volumes/NPC预测数据盘/单野预测2024/batch_test_h5/data_recoder.csv", header=None)
# h5_name_list = df.iloc[:, 0]
# id_list = df.iloc[:, 1]
# min_z_list = df.iloc[:, 2]
# max_z_list = df.iloc[:, 3]
# dim_z_list = df.iloc[:, 4]
# print(df)
beam_cross = [[0.272,   0.428,	0.786,	0.305,	0.455,  0.825],
              [0.257,	0.627,	0.695,	0.271,	0.616,	0.736],
              [0.358,	0.737,	0.734,	0.387,	0.728,	0.740],
              [0.313,	0.578,	0.684,	0.349,	0.592,	0.705],
              [0.119,	0.392,	0.713,	0.174,	0.400,	0.733],
              [0.139,	0.356,	0.709,	0.194,	0.372,	0.725],
              [0.325,	0.539,	0.692,	0.370,	0.551,	0.715],
              [0.368,	0.629,	0.720,	0.403,	0.647,	0.725],
              [0.260,	0.617,	0.701,	0.262,	0.608,	0.726]]

beam_line = [[0.295,	0.765,	0.438,	0.277,	0.754,	0.438],
             [0.308,	0.805,	0.273,	0.289,	0.793,	0.270],
             [0.236,	0.802,	0.206,	0.230,	0.790,	0.217],
             [0.261,	0.763,	0.217,	0.249,	0.734,	0.229],
             [0.265,	0.724,	0.268,	0.256,	0.692,	0.279],
             [0.279,	0.727,	0.276,	0.271,	0.699,	0.284],
             [0.271,	0.764,	0.219,	0.260,	0.731,	0.231],
             [0.246,	0.795,	0.205,	0.244,	0.786,	0.219],
             [0.327,	0.799,	0.286,	0.314,	0.790,	0.285]]

beam_cross = np.array(beam_cross)
beam_line = np.array(beam_line)
plt.rc('font', family='Times New Roman')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# 示例数据
pred_cross = beam_cross[:, 0:3]
pred_line = beam_line[:, 0:3]
# pred_cross = beam_cross[:, 3:]
# pred_line = beam_line[:, 3:]

input_columns = ['Input_CT', 'Input_PDD', 'Input_Projections']
# input_columns = ['Input_CT', 'Input_PDD', 'Input_Projections', 'Input_CT', 'Input_PDD', 'Input_Projections']

output_columns = ['Output_0°','Output_40°', 'Output_80°', 'Output_120°',
                  'Output_160°', 'Output_200°', 'Output_240°', 'Output_280°', 'Output_320°']

output_columns_gt = ['GT_0°', 'GT_40°', 'GT_80°', 'GT_120°', 'GT_160°', 'GT_200°', 'GT_240°', 'GT_280°', 'GT_320°']
font4 = {'family' : 'Times New Roman',
        # 'color'  : 'black',
        'weight' : 'bold',
        'size'   : 15,
        }
# 创建热力图
sns.set_theme(font_scale=1.2)
plt.figure(figsize=(5,8))
# h = sns.heatmap(pred_cross, xticklabels=input_columns, yticklabels=output_columns, cmap='coolwarm', annot=True, vmin=0, vmax=0.8,
#                 cbar=False, square=True)
# h = sns.heatmap(pred_cross, xticklabels=input_columns, yticklabels=output_columns_gt, cmap='coolwarm', annot=True, vmin=0, vmax=0.8,
#                 cbar=False, square=True)

# h = sns.heatmap(pred_line, xticklabels=input_columns, yticklabels=output_columns, cmap='coolwarm', annot=True, vmin=0, vmax=0.8,
#                 cbar=False, square=True)
h = sns.heatmap(pred_line, xticklabels=input_columns, yticklabels=output_columns_gt, cmap='coolwarm', annot=True, vmin=0, vmax=0.8,
                cbar=False, square=True)
plt.xticks(rotation=45, font=font4)
# cbar = h.figure.colorbar(h.collections[0])
# cbar.ax.tick_params(labelsize=15, color='darkred')
h.yaxis.set_ticks_position('right')
plt.yticks(font=font4, rotation=0)
plt.rcParams['savefig.dpi'] = 600
plt.tight_layout()
# plt.savefig('/Volumes/NPC预测数据盘/单野预测2024/line_spearmann_numbers/line_pred.jpg')
plt.savefig('/Volumes/NPC预测数据盘/单野预测2024/line_spearmann_numbers/line_gt.jpg')
# plt.savefig('/Volumes/NPC预测数据盘/单野预测2024/line_spearmann_numbers/cb.jpg')

# plt.show()