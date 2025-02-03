import h5py as h5
import hydra
import numpy as np
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'
import matplotlib.pyplot as plt
import pandas as pd
import einops


def max_dose_calculator(dose, mask):
    if mask.any() == False:
        return None
    points = dose * mask
    return "{:.3f}".format(points.max())


def min_dose_calculator(dose, mask):
    if mask.any() == False:
        return None
    points = dose * mask
    return "{:.3f}".format(np.min(points[np.where(points > 0)]))


def d95_calculator(dose, mask):
    if mask.any() == False:
        return None
    masked_dose = dose * mask * 10
    total_points = np.count_nonzero(masked_dose)
    # print("total points:{}".format(total_points))
    max_dose = int(np.ceil(np.max(masked_dose)))
    min_dose = int(np.floor(np.min(masked_dose[np.where(masked_dose > 0)])))
    for i in range(min_dose, max_dose):
        # print(i)
        count_i = np.count_nonzero(masked_dose > i)
        # print("count points:{}".format(count_i))
        precentage = count_i / total_points
        # print("per{}".format(precentage))
        if precentage < 0.95:
            print(i)
            break

    return "{:.3f}".format(i / 10)


def d5_calculator(dose, mask):
    if mask.any() == False:
        return None
    masked_dose = dose * mask * 10
    total_points = np.count_nonzero(masked_dose)
    # print("total points:{}".format(total_points))

    max_dose = int(np.ceil(np.max(masked_dose)))
    min_dose = int(np.floor(np.min(masked_dose[np.where(masked_dose > 0)])))
    for i in range(min_dose, max_dose):
        # print(i)
        count_i = np.count_nonzero(masked_dose > i)
        precentage = count_i / total_points
        # print("count points:{}".format(count_i))
        if precentage < 0.05:
            print(i)
            break

    return "{:.3f}".format(i / 10)


def dmean_calculator(dose, mask):
    if mask.any() == False:
        return None
    ind = np.where(mask == True)
    return "{:.3f}".format(dose[ind].mean())


def CI_calculator(dose, mask, prescription):
    if mask.any() == False:
        return None
    vri_ind = dose > prescription
    V_RI = np.count_nonzero(vri_ind)
    TV = np.count_nonzero(mask)
    tv_ind = vri_ind * mask
    TV_RI = np.count_nonzero(tv_ind)

    return "{:.3f}".format((TV_RI * TV_RI) / (V_RI * TV))

"""z左右腮腺的名字可能还要改下，匹配所有的name。"""
df = pd.read_csv("/Volumes/NPC预测数据盘/单野预测2024/batch_test_h5/data_recoder.csv", dtype=str, header=None)
h5_list = df.iloc[:, 0]
pt_id = df.iloc[:, 1]
min_range = df.iloc[:, 3]
max_range = df.iloc[:, 4]


with open("/Volumes/NPC预测数据盘/单野预测2024/DVH_table/PPIM-BDMG.csv", 'w+') as csv_f:
    csv_f.write('ptv70_dmax, ptv70_dmin, ptv70_d95, ptv70_d5, ptv70_ci,'
                ' ptv60_dmax, ptv60_dmin, ptv60_d95, ptv60_d5, ptv60_ci,'
                'ptv54_dmax, ptv54_dmin, ptv54_d95, ptv54_d5, ptv54_ci, '
                'cord_dmax, brainstem_dmax, optic_L_dmax, optic_R_dmax,'
                'chiasm_dmax, parotid_L_dmean, parotid_R_dmean, larynx_dmean,')
    csv_f.write('\n')
    # csv_f.write(',')
dir_path = "/Volumes/NPC预测数据盘/单野预测2024/pred_dose/PPIM-BDMG"
for index in range(len(df)):
    print(pt_id[index])
    h5_name = "/Volumes/NPC预测数据盘/单野预测2024/batch_test_h5/" + h5_list[index] + '.h5'
    f = h5.File(h5_name, 'r')
    min_z = int(min_range[index])
    max_z = int(max_range[index])

    pred_path = dir_path + os.sep +  pt_id[index] + "_-1.npy"
    pred_dose = np.load(pred_path)
    real_dose = pred_dose[:, :, min_z:max_z, -1]

    mask_npy = f["slice_mask"][:, :, min_z:max_z, :]
    # real_dose = f["slice_dose"][:, :, :, -1]
    ptv70 = mask_npy[:, :, :, 1]
    ptv60 = mask_npy[:, :, :, 2]
    ptv54 = mask_npy[:, :, :, 3]
    brain = mask_npy[:, :, :, 4]
    cord = mask_npy[:, :, :, 5]
    left_parotid = mask_npy[:, :, :, 6]
    right_parotid = mask_npy[:, :, :, 7]
    chiasm = mask_npy[:, :, :, 8]
    optic_nerve_l = mask_npy[:, :, :, 9]
    optic_nerve_r = mask_npy[:, :, :, 10]
    larynx = mask_npy[:, :, :, -1]
    ptv70_max = max_dose_calculator(real_dose, ptv70)
    ptv70_min = min_dose_calculator(real_dose, ptv70)
    ptv70_d95 = d95_calculator(real_dose, ptv70)
    ptv70_d5 = d5_calculator(real_dose, ptv70)
    ptv70_ci = CI_calculator(real_dose, ptv70, 70)

    ptv60_max = max_dose_calculator(real_dose, ptv60)
    ptv60_min = min_dose_calculator(real_dose, ptv60)
    ptv60_d95 = d95_calculator(real_dose, ptv60)
    ptv60_d5 = d5_calculator(real_dose, ptv60)
    ptv60_ci = CI_calculator(real_dose, ptv60, 60)

    ptv54_max = max_dose_calculator(real_dose, ptv54)
    ptv54_min = min_dose_calculator(real_dose, ptv54)
    ptv54_d95 = d95_calculator(real_dose, ptv54)
    ptv54_d5 = d5_calculator(real_dose, ptv54)
    ptv54_ci = CI_calculator(real_dose, ptv54, 54)

    cord_max = max_dose_calculator(real_dose, cord)
    bs_max = max_dose_calculator(real_dose, brain)
    OL = max_dose_calculator(real_dose, optic_nerve_l)
    OR = max_dose_calculator(real_dose, optic_nerve_r)

    chiasm_max = max_dose_calculator(real_dose, chiasm)
    PL = dmean_calculator(real_dose, left_parotid)
    PR = dmean_calculator(real_dose, right_parotid)
    larynx_mean = dmean_calculator(real_dose, larynx)

    with open("/Volumes/NPC预测数据盘/单野预测2024/DVH_table/PPIM-BDMG.csv", 'a+') as csv_file:
        csv_file.write("{},{},{},{},{},"
                       "{},{},{},{},{},"
                       "{},{},{},{},{},"
                       "{},{},{},{},"
                       "{},{},{},{},".format(ptv70_max, ptv70_min, ptv70_d95, ptv70_d5, ptv70_ci,
                                            ptv60_max, ptv60_min, ptv60_d95, ptv60_d5, ptv60_ci,
                                            ptv54_max, ptv54_min, ptv54_d95, ptv54_d5, ptv54_ci,
                                            cord_max, bs_max, OL, OR,
                                            chiasm_max, PL, PR, larynx_mean))
        csv_file.write('\n')

