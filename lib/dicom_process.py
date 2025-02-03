import logging
import multiprocessing
import os
import shutil
import uuid
from itertools import repeat

import cv2
import h5py
import numpy as np

from lib.color_palette import *
from lib.dicom.dicom_directory import DicomDirectory
from lib.dicom.dicom_dose_series import DicomDoseSeries
from lib.dicom.dicom_image_series import DicomCTSeries
from lib.dicom.dicom_image_series import DicomMRSeries
from lib.dicom.dicom_rtst import DicomRTST
from lib.utilities import clean_folder
from lib.utilities import random_path, dataset_split, dataset_cv_split
from lib.utilities import shape_check_and_rescale
from lib.utilities import to_abs_path
from seg_task.Beams_Contour_Generator import BeamContourGeneration
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


# === block 1. rtst training and validation data === start
def generate_rtst_train_validation_data(cfg):
    dicom_path = to_abs_path(cfg.dicom_path)
    train_path = to_abs_path(cfg.train_path)
    validation_path = to_abs_path(cfg.validation_path)

    dicomdir = DicomDirectory(dicom_path)
    dicomdir.scan()
    clean_folder(train_path)
    clean_folder(validation_path)

    logger.info('start create training and validation hdf5 dataset.')
    if 'sort_method' in cfg:
        logger.info('sort method: ' + cfg.sort_method)
    else:
        logger.info('default sort method: instance number')

    if cfg.thread_num == 1:
        for dicom_rtst_series, img_series_info in dicomdir.rtst_and_its_img_series_iter():
            output_path = random_path(cfg)
            _rtst_train_validation_data(dicom_rtst_series, img_series_info, cfg, output_path)
    else:
        with multiprocessing.Pool(processes=cfg.thread_num) as pool:
            rtst_list = [rtst for rtst, img in dicomdir.rtst_and_its_img_series_iter()]
            img_list = [img for rtst, img in dicomdir.rtst_and_its_img_series_iter()]
            output_path_list = [random_path(cfg) for i in range(len(img_list))]
            pool.starmap(_rtst_train_validation_data, zip(rtst_list, img_list, repeat(cfg), output_path_list))


def _rtst_train_validation_data(dicom_rtst_series, img_series_info, cfg, output_path):
    img_series = DicomCTSeries(img_series_info)
    if 'sort_method' in cfg:
        img_series.set_sort_method(cfg.sort_method)
    img_series.load_data()

    rtst_file = dicom_rtst_series.single_file
    rtst_dataset = DicomRTST(rtst_file)
    pt_id = rtst_dataset.PatientID

    if 'exclude_pt_list' in cfg:
        pass
        # todo: exclude some patient

    # normalization setting
    if cfg.norm_method == 'fix':
        img_series.norm_method = 'fix'
        img_series.min_value = cfg.norm_low
        img_series.max_value = cfg.norm_high

    slice_img = img_series.normalized_voxel_array

    if slice_img is None:
        logger.warning('slice_img error for [{}].'.format(pt_id))
        return None

    roi_dicts = cfg.roi_dicts

    slice_mask = np.zeros(list(slice_img.shape) + [len(roi_dicts)], dtype=int)
    all_roi_exist = rtst_dataset.roi_exist_check_and_match(roi_dicts)

    if (not all_roi_exist) and cfg.remove_data_with_problems:
        logger.warning('not generate h5py file for [{}].'.format(pt_id))
        return None

    for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):
        roi_mask, success_flag = rtst_dataset.create_3d_mask(roi_ind_in_conf, img_series, use_4d=True)
        slice_mask[:, :, :, roi_ind_in_conf] = roi_mask

    slice_img = shape_check_and_rescale(slice_img, cfg.dim_y, cfg.dim_x)
    slice_mask = shape_check_and_rescale(slice_mask, cfg.dim_y, cfg.dim_x)

    p_roi = cal_roi_balance_table(slice_mask)

    data_id_str = str(uuid.uuid4())[:8]
    # output_path = random_path(cfg)

    with h5py.File(output_path + os.sep + data_id_str + ".h5", "w") as hf:
        hf.create_dataset("slice_img", data=slice_img, compression='lzf')
        hf.create_dataset("slice_mask", data=slice_mask, compression='lzf')
        hf.create_dataset("p_roi", data=p_roi, compression='lzf')

    with open(output_path + os.sep + "data_recoder.csv", 'a+') as csv_file:
        csv_file.write(data_id_str + ',' + pt_id + ',' +
                       str(img_series.z_dim) + ',' + '\n')
    logger.info('[{}] h5py file generate success.'.format(pt_id))


# ===  rtst training and validation data === end


# === block 2. ct training and validation data === start
def generate_ct_train_validation_data(cfg):
    dicom_path = to_abs_path(cfg.dicom_path)
    train_path = to_abs_path(cfg.train_path)
    validation_path = to_abs_path(cfg.validation_path)

    clean_folder(train_path)
    clean_folder(validation_path)

    if not isinstance(dicom_path, list):
        dicom_path_list = [dicom_path]
    else:
        dicom_path_list = dicom_path

    for dicom_path in dicom_path_list:
        dicomdir = DicomDirectory(dicom_path)
        dicomdir.scan()

        logger.info('start create training and validation hdf5 dataset.')
        if cfg.thread_num == 1:
            for img_series_info in dicomdir.series_iter(series_type='CT Image Storage'):
                output_path = random_path(cfg)
                _ct_train_validation_data(img_series_info, cfg, output_path)
        else:
            with multiprocessing.Pool(processes=cfg.thread_num) as pool:
                img_list = [img for img in dicomdir.series_iter(series_type='CT Image Storage')]
                output_path_list = [random_path(cfg) for i in range(len(img_list))]
                pool.starmap(_ct_train_validation_data, zip(img_list, repeat(cfg), output_path_list))


def _ct_train_validation_data(img_series_info, cfg, output_path):
    img_series = DicomCTSeries(img_series_info)
    if 'sort_method' in cfg:
        img_series.set_sort_method(cfg.sort_method)
    img_series.load_data()

    pt_id = img_series.PatientID

    if 'exclude_pt_list' in cfg:
        pass
        # todo: exclude some patient

    # normalization setting
    if cfg.norm_method == 'fix':
        img_series.norm_method = 'fix'
        img_series.min_value = cfg.norm_low
        img_series.max_value = cfg.norm_high

    slice_img = img_series.normalized_voxel_array
    slice_img = shape_check_and_rescale(slice_img, cfg.dim_y, cfg.dim_x)

    data_id_str = str(uuid.uuid4())[:8]

    with h5py.File(output_path + os.sep + data_id_str + ".h5", "w") as hf:
        hf.create_dataset("slice_img", data=slice_img, compression='lzf')

    with open(output_path + os.sep + "data_recoder.csv", 'a+') as csv_file:
        csv_file.write(data_id_str + ',' + pt_id + ',' +
                       str(img_series.z_dim) + ',' + '\n')
    logger.info('[{}] h5py file generate success.'.format(pt_id))


# === ct training and validation data === end


# === block 3, generate rtst check images === start
def generate_rtst_check_images(cfg):
    dicom_path = to_abs_path(cfg.dicom_path)
    dicomdir = DicomDirectory(dicom_path)
    dicomdir.scan()

    check_img_path = to_abs_path(cfg.check_img_path)
    clean_folder(check_img_path)

    logger.info('start create rtst check images.')
    if cfg.thread_num == 1:
        for dicom_rtst_series, img_series_info in dicomdir.rtst_and_its_img_series_iter():
            _rtst_check_image(dicom_rtst_series, img_series_info, cfg, check_img_path)
    else:
        with multiprocessing.Pool(processes=cfg.thread_num) as pool:
            rtst_list = [rtst for rtst, img in dicomdir.rtst_and_its_img_series_iter()]
            img_list = [img for rtst, img in dicomdir.rtst_and_its_img_series_iter()]
            pool.starmap(_rtst_check_image, zip(rtst_list, img_list, repeat(cfg), repeat(check_img_path)))


def _rtst_check_image(dicom_rtst_series, img_series_info, cfg, output_path):
    img_series = DicomCTSeries(img_series_info)
    if 'sort_method' in cfg:
        img_series.set_sort_method(cfg.sort_method)
    img_series.load_data()

    rtst_file = dicom_rtst_series.single_file
    rtst_dataset = DicomRTST(rtst_file)
    pt_id = rtst_dataset.PatientID

    if 'exclude_pt_list' in cfg:
        pass
        # todo: exclude some patient

    check_imgs = img_series.check_image_soft

    roi_dicts = cfg.roi_dicts
    all_roi_exist = rtst_dataset.roi_exist_check_and_match(roi_dicts)

    for z in range(check_imgs.shape[2]):

        combine_check_img_rgb = None
        check_img = np.array(check_imgs[:, :, z], dtype=np.uint8)
        check_img_rgb = cv2.cvtColor(check_img, cv2.COLOR_GRAY2RGB)

        for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):

            roi_check_imgs = rtst_dataset.generate_check_roi_mask(roi_ind_in_conf, img_series)
            roi_color = set1[roi_ind_in_conf]

            roi_check_img = roi_check_imgs[:, :, z]
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
                png_file_name = output_path + os.sep + "roi_[" + pt_id + "]_[" + str(roi_ind_in_conf) + "]_z_[" + \
                                str(z) + "].png"
                cv2.imwrite(png_file_name, roi_check_img_rgb)

        png_file_name = output_path + os.sep + "pt_id_[" + pt_id + "]_z_[" + str(z) + "].png"
        cv2.imwrite(png_file_name, combine_check_img_rgb)


# === rtst check images === end


# === block 4, rtst glance images === start
def generate_rtst_glance_images(cfg):
    dicom_path = to_abs_path(cfg.dicom_path)
    dicomdir = DicomDirectory(dicom_path)
    dicomdir.scan()

    check_img_path = to_abs_path(cfg.check_img_path)
    clean_folder(check_img_path)

    logger.info('start create glance images.')
    if 'sort_method' in cfg:
        logger.info('sort method: ' + cfg.sort_method)
    else:
        logger.info('default sort method: instance number')

    if cfg.thread_num == 1:
        for dicom_rtst_series, img_series_info in dicomdir.rtst_and_its_img_series_iter():
            _rtst_glance_image(dicom_rtst_series, img_series_info, cfg, check_img_path)
    else:
        with multiprocessing.Pool(processes=cfg.thread_num) as pool:
            rtst_list = [rtst for rtst, img in dicomdir.rtst_and_its_img_series_iter()]
            img_list = [img for rtst, img in dicomdir.rtst_and_its_img_series_iter()]
            pool.starmap(_rtst_glance_image, zip(rtst_list, img_list, repeat(cfg), repeat(check_img_path)))


def _rtst_glance_image(dicom_rtst_series, img_series_info, cfg, output_path):
    img_series = DicomCTSeries(img_series_info)
    if 'sort_method' in cfg:
        img_series.set_sort_method(cfg.sort_method)

    img_series.load_data()

    rtst_file = dicom_rtst_series.single_file
    rtst_dataset = DicomRTST(rtst_file)
    pt_id = rtst_dataset.PatientID

    if 'exclude_pt_list' in cfg:
        pass
        # todo: exclude some patient

    glance_img = img_series.glance_image_soft
    glance_img_rgb = cv2.cvtColor(np.array(glance_img, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

    roi_dicts = cfg.roi_dicts
    all_roi_exist = rtst_dataset.roi_exist_check_and_match(roi_dicts)

    combine_glance_img_rgb = None
    for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):

        roi_color = set1[roi_ind_in_conf]

        roi_glance_image = rtst_dataset.generate_glance_roi_mask(roi_ind_in_conf, img_series)
        roi_glance_image_rgb = np.repeat(roi_glance_image[:, :, np.newaxis], 3, axis=2)
        roi_glance_image_rgb = roi_glance_image_rgb * np.array(roi_color, dtype=int)
        roi_glance_image_rgb = np.array(roi_glance_image_rgb, dtype=np.uint8)

        if combine_glance_img_rgb is None:
            combine_glance_img_rgb = cv2.addWeighted(glance_img_rgb, 0.5, roi_glance_image_rgb,
                                                     cfg.roi_color_overlay_alpha, 0)
        else:
            combine_glance_img_rgb = cv2.addWeighted(combine_glance_img_rgb, 1.0, roi_glance_image_rgb,
                                                     cfg.roi_color_overlay_alpha, 0)

        if cfg.gen_mask_img_for_each_roi:
            png_file_name = output_path + os.sep + "roi_[" + pt_id + "]_[" + str(roi_ind_in_conf) + "].png"
            cv2.imwrite(png_file_name, roi_glance_image_rgb)
    # for roi in roi_dicts:
    #     print(roi)
    png_file_name = output_path + os.sep + "pt_id_[" + pt_id + "].png"
    cv2.imwrite(png_file_name, combine_glance_img_rgb)


# === rtst glance images === end


# === block 5. ct glance images === start
def generate_ct_glance_images(cfg):
    dicom_path = to_abs_path(cfg.dicom_path)
    dicomdir = DicomDirectory(dicom_path)
    dicomdir.scan()

    check_img_path = to_abs_path(cfg.check_img_path)
    clean_folder(check_img_path)

    logger.info('start create glance images.')
    if 'sort_method' in cfg:
        logger.info('sort method: ' + cfg.sort_method)
    else:
        logger.info('default sort method: instance number')
    if cfg.thread_num == 1:
        for img_series_info in dicomdir.series_iter(series_type='CT Image Storage'):
            _ct_glance_image(img_series_info, cfg, check_img_path)
    else:
        with multiprocessing.Pool(processes=cfg.thread_num) as pool:
            img_list = [img for img in dicomdir.series_iter(series_type='CT Image Storage')]
            pool.starmap(_ct_glance_image, zip(img_list, repeat(cfg), repeat(check_img_path)))


def _ct_glance_image(img_series_info, cfg, output_path):
    img_series = DicomCTSeries(img_series_info)
    if 'sort_method' in cfg:
        img_series.set_sort_method(cfg.sort_method)
    img_series.load_data()

    pt_id = img_series.PatientID
    series_uid = img_series.SeriesInstanceUID

    if 'exclude_pt_list' in cfg:
        pass
        # todo: exclude some patient
    try:
        glance_img = img_series.glance_image_soft
        glance_img_rgb = cv2.cvtColor(np.array(glance_img, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        png_file_name = output_path + os.sep + "pt_id_[" + pt_id + "]_uid_[" + series_uid + "].png"
        cv2.imwrite(png_file_name, glance_img_rgb)
    except:
        logger.info('error in patient: ' + pt_id)


# === image glance images === end


# === block 6, rtst mr glance images === start
def generate_mr_rtst_glance_images(cfg):
    dicom_path = to_abs_path(cfg.dicom_path)
    dicomdir = DicomDirectory(dicom_path)
    dicomdir.scan()

    check_img_path = to_abs_path(cfg.check_img_path)
    clean_folder(check_img_path)

    logger.info('start create glance images.')

    series_info_list = None
    if 'multi_series' in cfg:
        if cfg.multi_series:
            series_info_list = cfg.series_dicts

    if cfg.thread_num == 1:
        for dicom_rtst_series, img_series_info in dicomdir.rtst_and_its_img_series_iter(
                series_info_list=series_info_list):
            _mr_rtst_glance_image(dicom_rtst_series, img_series_info, cfg, check_img_path)
    else:
        with multiprocessing.Pool(processes=cfg.thread_num) as pool:
            rtst_list = [rtst for rtst, img in dicomdir.rtst_and_its_img_series_iter()]
            img_list = [img for rtst, img in dicomdir.rtst_and_its_img_series_iter(series_info_list=series_info_list)]
            pool.starmap(_mr_rtst_glance_image, zip(rtst_list, img_list, repeat(cfg), repeat(check_img_path)))


def _mr_rtst_glance_image(dicom_rtst_series, img_series_info, cfg, output_path):
    rtst_file = dicom_rtst_series.single_file
    rtst_dataset = DicomRTST(rtst_file)
    pt_id = rtst_dataset.PatientID
    roi_dicts = cfg.roi_dicts
    _ = rtst_dataset.roi_exist_check_and_match(roi_dicts)  # do not delete

    if isinstance(img_series_info, list):
        # establish the slice relation shape between image series
        # dict: mian_ref_uid: [loc, uid1, uid2, ...]
        slice_location_map = dict()
        img_series_list = []

        for img_series_info_i in img_series_info:
            img_series_i = DicomMRSeries(img_series_info_i)
            img_series_i.load_data()
            img_series_list.append(img_series_i)

            # establish slice location map for first series
            if len(slice_location_map) == 0:
                for slice_uid, loc in img_series_i.ImagePositionPatient.items():
                    slice_location_map[slice_uid] = [loc[2]]
            else:
                # map to other series
                all_mapped_flag = True
                for uid_main, loc_main in slice_location_map.items():
                    mapped_flag = False
                    for slice_uid, loc in img_series_i.ImagePositionPatient.items():
                        if abs(loc_main[0] - loc[2]) < 0.01:
                            slice_location_map[uid_main].append(slice_uid)
                            mapped_flag = True
                            break
                    all_mapped_flag = all_mapped_flag and mapped_flag
                # not all mapped
                if not all_mapped_flag:
                    logger.warning('[{}] image incomplete.'.format(pt_id))
                    return None

        if 'exclude_pt_list' in cfg:
            pass
            # todo: exclude some patient

        for img_series_num, img_series_i in enumerate(img_series_list):
            if img_series_num == 0:
                glance_img = img_series_i.glance_image
                sorted_main_series_uid = img_series_i.sorted_uid_list
            else:
                attached_uid_list = []
                # get the 2nd series' uid list
                for uid in sorted_main_series_uid:
                    attached_uid_list.append(slice_location_map[uid][img_series_num])

                glance_img = img_series_i.gen_glance_image_with_uid_list(attached_uid_list)

            glance_img_rgb = cv2.cvtColor(np.array(glance_img, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

            combine_glance_img_rgb = None
            for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):

                roi_color = set1[roi_ind_in_conf]

                # only for main series
                if img_series_num == 0:
                    roi_glance_image = rtst_dataset.generate_glance_roi_mask(roi_ind_in_conf, img_series_i)
                else:
                    roi_glance_image = rtst_dataset.generate_glance_roi_mask(roi_ind_in_conf, img_series_list[0])

                roi_glance_image_rgb = np.repeat(roi_glance_image[:, :, np.newaxis], 3, axis=2)
                roi_glance_image_rgb = roi_glance_image_rgb * np.array(roi_color, dtype=int)
                roi_glance_image_rgb = np.array(roi_glance_image_rgb, dtype=np.uint8)

                if combine_glance_img_rgb is None:
                    glance_img_rgb = shape_check_and_rescale(glance_img_rgb,
                                                             roi_glance_image_rgb.shape[0],
                                                             roi_glance_image_rgb.shape[1])
                    combine_glance_img_rgb = cv2.addWeighted(glance_img_rgb, 0.5, roi_glance_image_rgb,
                                                             cfg.roi_color_overlay_alpha, 0)
                else:
                    combine_glance_img_rgb = cv2.addWeighted(combine_glance_img_rgb, 1.0, roi_glance_image_rgb,
                                                             cfg.roi_color_overlay_alpha, 0)

                if cfg.gen_mask_img_for_each_roi:
                    png_file_name = output_path + os.sep + "roi_[" + pt_id + "]_[" + str(roi_ind_in_conf) + "].png"
                    cv2.imwrite(png_file_name, roi_glance_image_rgb)

            png_file_name = output_path + os.sep + "pt_id_[" + pt_id + "]series[" + str(img_series_num) + "].png"
            cv2.imwrite(png_file_name, combine_glance_img_rgb)
    else:
        img_series = DicomMRSeries(img_series_info)
        img_series.load_data()

        rtst_file = dicom_rtst_series.single_file
        rtst_dataset = DicomRTST(rtst_file)
        pt_id = rtst_dataset.PatientID

        if 'exclude_pt_list' in cfg:
            pass
            # todo: exclude some patient

        glance_img = img_series.glance_image
        glance_img_rgb = cv2.cvtColor(np.array(glance_img, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

        roi_dicts = cfg.roi_dicts

        combine_glance_img_rgb = None
        for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):

            roi_color = set1[roi_ind_in_conf]

            roi_glance_image = rtst_dataset.generate_glance_roi_mask(roi_ind_in_conf, img_series)
            roi_glance_image_rgb = np.repeat(roi_glance_image[:, :, np.newaxis], 3, axis=2)
            roi_glance_image_rgb = roi_glance_image_rgb * np.array(roi_color, dtype=int)
            roi_glance_image_rgb = np.array(roi_glance_image_rgb, dtype=np.uint8)

            if combine_glance_img_rgb is None:
                combine_glance_img_rgb = cv2.addWeighted(glance_img_rgb, 0.5, roi_glance_image_rgb,
                                                         cfg.roi_color_overlay_alpha, 0)
            else:
                combine_glance_img_rgb = cv2.addWeighted(combine_glance_img_rgb, 1.0, roi_glance_image_rgb,
                                                         cfg.roi_color_overlay_alpha, 0)

            if cfg.gen_mask_img_for_each_roi:
                png_file_name = output_path + os.sep + "roi_[" + pt_id + "]_[" + str(roi_ind_in_conf) + "].png"
                cv2.imwrite(png_file_name, roi_glance_image_rgb)

        png_file_name = output_path + os.sep + "pt_id_[" + pt_id + "].png"
        cv2.imwrite(png_file_name, combine_glance_img_rgb)


# === rtst mr glance images === end



# === block 8. dose, rtst training and validation data === start
def gen_dose_rtst_train_validation_data(cfg):
    dicom_path = to_abs_path(cfg.dicom_path)
    train_path = to_abs_path(cfg.train_path)
    validation_path = to_abs_path(cfg.validation_path)

    dicomdir = DicomDirectory(dicom_path)
    dicomdir.scan()
    clean_folder(train_path)
    clean_folder(validation_path)

    logger.info('start create training and validation hdf5 dataset.')
    if 'sort_method' in cfg:
        logger.info('sort method: ' + cfg.sort_method)
    else:
        logger.info('default sort method: instance number')

    if cfg.thread_num == 1:
        for dicom_dose_series, dicom_rtst_series, img_series_info in dicomdir.dose_rtst_and_its_img_series_iter():
            output_path = random_path(cfg)
            _dose_rtst_train_validation_data(dicom_dose_series, dicom_rtst_series, img_series_info, cfg, output_path)
    else:
        with multiprocessing.Pool(processes=cfg.thread_num) as pool:
            dose_list = [dose for dose, rtst, img in dicomdir.dose_rtst_and_its_img_series_iter()]
            rtst_list = [rtst for dose, rtst, img in dicomdir.dose_rtst_and_its_img_series_iter()]
            img_list = [img for dose, rtst, img in dicomdir.dose_rtst_and_its_img_series_iter()]
            output_path_list = [random_path(cfg) for i in range(len(img_list))]
            pool.starmap(_dose_rtst_train_validation_data,
                         zip(dose_list, rtst_list, img_list, repeat(cfg), output_path_list))


def _dose_rtst_train_validation_data(dicom_dose_series, dicom_rtst_series, img_series_info, cfg, output_path):
    img_series = DicomCTSeries(img_series_info)
    if 'sort_method' in cfg:
        img_series.set_sort_method(cfg.sort_method)
    img_series.load_data()

    rtst_file = dicom_rtst_series.single_file
    rtst_dataset = DicomRTST(rtst_file)
    pt_id = rtst_dataset.PatientID
    patient_directory, patient_file = os.path.split(rtst_file)

    if 'exclude_pt_list' in cfg:
        pass
        # todo: exclude some patient

    # normalization setting
    if cfg.norm_method == 'fix':
        img_series.norm_method = 'fix'
        img_series.min_value = cfg.norm_low
        img_series.max_value = cfg.norm_high

    slice_img = img_series.normalized_voxel_array
    roi_dicts = cfg.roi_dicts
    raw_z_dim = img_series.z_dim

    dose_dataset = DicomDoseSeries(dicom_dose_series)
    dicomdir = DicomDirectory(patient_directory)
    dicomdir.scan()
    slice_dose = np.zeros(list(slice_img.shape) + [10])

    # beam_doses = np.zeros((slice_img.shape[0], slice_img.shape[1], slice_img.shape[2]))
    # 这里dir不必scan全部文件，只扫描dose series的上级目录就可以了。
    for dose_uid, dose_file in dicomdir.file_iter(series_type="RT Dose Storage", patient_id=pt_id):
        beam_index = dose_file[-5]
        dose_dataset.load_data_by_file_name(dose_file)
        beam_dose = dose_dataset.to_img_voxel(img_series)
        if beam_index == 'e':
            slice_dose[:, :, :, -1] = beam_dose
            # break
            continue
        else:
            slice_dose[:, :, :, int(beam_index) - 1] = beam_dose


    # dose_dataset.load_data()
    logger.info("start to create masks")
    slice_mask = np.zeros(list(slice_img.shape) + [len(roi_dicts)], dtype=bool)
    # slice_mask = np.zeros(list(slice_img.shape) + [len(roi_dicts)])

    all_roi_exist = rtst_dataset.roi_exist_check_and_match(roi_dicts)

    if (not all_roi_exist) and cfg.remove_data_with_problems:
        logger.warning('not generate h5py file for [{}].'.format(pt_id))
        return None

    for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):
        roi_mask, success_flag = rtst_dataset.create_3d_mask(roi_ind_in_conf, img_series)
        slice_mask[:, :, :, roi_ind_in_conf] = roi_mask
    logger.info("roi masks has created!")
    ptvs = np.sum(slice_mask[:, :, :, 1:4], axis=-1, keepdims=False)
    z_min = np.nonzero(ptvs)[-1].min()
    z_max = np.nonzero(ptvs)[-1].max()

    # slice_img = shape_check_and_rescale(slice_img, cfg.dim_y, cfg.dim_x)
    # slice_mask = shape_check_and_rescale(slice_mask, cfg.dim_y, cfg.dim_x)
    # slice_dose = shape_check_and_rescale(slice_dose, cfg.dim_y, cfg.dim_x)
    # for zi in range(raw_z_dim):
    #     if slice_dose[:, :, zi, -1].max() > cfg.dose_clip_threshold:
    #         break
    # for zj in range(raw_z_dim):
    #     if slice_dose[:, :, raw_z_dim - zj - 1, -1].max() > cfg.dose_clip_threshold:
    #         break

    # slice_dose = slice_dose[:, :, zi:raw_z_dim - zj -1, :]
    slice_dose = slice_dose[:, :, z_min:(z_max + 1), :]
    img = slice_img[:, :, z_min:(z_max + 1)]
    mask = slice_mask[:, :, z_min:(z_max + 1), :]

    # slice_img = slice_img[:, :, zi:raw_z_dim - zj -1]
    # slice_mask = slice_mask[:, :, zi:raw_z_dim - zj -1, :]
    z_dim = slice_dose.shape[2]
    body = mask[:, :, :, 0:1]
    dose = slice_dose * body

    # img = zoom(slice_img, (0.5, 0.5, 1), order=0)
    # mask = zoom(slice_mask, (0.5, 0.5, 1, 1), order=0)
    # dose = zoom(slice_dose, (0.5, 0.5, 1, 1), order=0)

    print("slice_img.shape:",img.shape)
    print("slice_mask.shape:",mask.shape)
    print("slice_dose.shape:", dose.shape)
    # print("slice_dose_max:", np.max(slice_dose[:, :, :, int(cfg.beam_index) - 1]))
    bc = BeamContourGeneration(mask, pt_id)
    beams= bc.output()
    # slice_beam = np.array(slice_beam, dtype=bool)
    """output_5d_numpy=[512, 512, z, 9, 13], shape(4)=0 代表0度下所有roi的beam contour。"""

    # comp_beam = np.logical_and(slice_beam, slice_mask[:, :, :, 0])
    # slice_mask[:, :, :, -1] = comp_beam
    # beam index: 0~8, 0=200degree, 1=240degree, ... 8=160 degree, slice mask留了一个channel给后面的beam contour预备，
    # 在seg_data里面调用。

    data_id_str = str(uuid.uuid4())[:8]
    # output_path = random_path(cfg)

    with h5py.File(output_path + os.sep + data_id_str + ".h5", "w") as hf:
        hf.create_dataset("slice_img", data=img, compression='lzf')
        # hf.create_dataset("slice_img", data=slice_img)

        hf.create_dataset("slice_mask", data=mask, compression='lzf')
        # hf.create_dataset("slice_mask", data=slice_mask)

        hf.create_dataset("slice_dose", data=dose, compression='lzf')
        # hf.create_dataset("slice_dose", data=slice_dose)

        hf.create_dataset("slice_beam", data=beams, compression='lzf')
        # hf.create_dataset("slice_beam", data=slice_beam)

        # hf.create_dataset("slice_grad", data=slice_grad, compression='lzf')
    #     dose:[512, 512, z, 10]; img[512, 512, z]; mask[512, 512, z, lenth_of_roi_dicts], beam[512, 512, z, 9]
    with open(output_path + os.sep + "data_recoder.csv", 'a+') as csv_file:
        csv_file.write(data_id_str + ',' + pt_id + ',' +
                       str(z_dim) + ',' + str(z_min) + ',' + str(z_max + 1) + ',' + '\n')
    logger.info('[{}] h5py file generate success.'.format(pt_id))
    hf.close()


# ===  dose, rtst training and validation data === end


def trim_dicom(dicom_path, des_path, series_type, anonymize=False):
    dicom_path = to_abs_path(dicom_path)
    des_path = to_abs_path(des_path)

    clean_folder(des_path)

    if not isinstance(dicom_path, list):
        dicom_path_list = [dicom_path]
    else:
        dicom_path_list = dicom_path

    for dicom_path in dicom_path_list:
        dicomdir = DicomDirectory(dicom_path)
        dicomdir.scan()

        logger.info('start trim data.')

        for dicom_series in dicomdir.series_iter(series_type):

            if anonymize:
                pass
            else:
                series_dir = des_path + os.sep + dicom_series.series_instance_uid
                os.mkdir(series_dir)
                for uid, dicom_file in dicom_series.items():
                    shutil.copy(dicom_file, series_dir + os.sep + uid + '.dcm')


# ========================block 9 create CT and dose data (without mask)===========================
def gen_dose_ct_train_validation_data(cfg):
    dicom_path = to_abs_path(cfg.dicom_path)
    train_path = to_abs_path(cfg.train_path)
    validation_path = to_abs_path(cfg.validation_path)

    dicomdir = DicomDirectory(dicom_path)
    dicomdir.scan()
    clean_folder(train_path)
    clean_folder(validation_path)

    logger.info('start create training and validation hdf5 dataset.')
    if 'sort_method' in cfg:
        logger.info('sort method: ' + cfg.sort_method)
    else:
        logger.info('default sort method: instance number')

    if cfg.thread_num == 1:
        for dicom_dose_series, dicom_rtst_series, img_series_info in dicomdir.dose_rtst_and_its_img_series_iter():
            output_path = random_path(cfg)
            _dose_ct_train_validation_data(dicom_dose_series, dicom_rtst_series, img_series_info, cfg, output_path)
    else:
        with multiprocessing.Pool(processes=cfg.thread_num) as pool:
            dose_list = [dose for dose, rtst, img in dicomdir.dose_rtst_and_its_img_series_iter()]
            rtst_list = [rtst for dose, rtst, img in dicomdir.dose_rtst_and_its_img_series_iter()]
            img_list = [img for dose, rtst, img in dicomdir.dose_rtst_and_its_img_series_iter()]
            output_path_list = [random_path(cfg) for i in range(len(img_list))]
            pool.starmap(_dose_ct_train_validation_data,
                         zip(dose_list, rtst_list, img_list, repeat(cfg), output_path_list))

def _dose_ct_train_validation_data(dicom_dose_series, dicom_rtst_series, img_series_info, cfg, output_path):
    img_series = DicomCTSeries(img_series_info)
    if 'sort_method' in cfg:
        img_series.set_sort_method(cfg.sort_method)
    img_series.load_data()

    rtst_file = dicom_rtst_series.single_file
    rtst_dataset = DicomRTST(rtst_file)
    pt_id = rtst_dataset.PatientID

    dose_dataset = DicomDoseSeries(dicom_dose_series)
    dose_dataset.load_data()
    slice_dose = dose_dataset.to_img_voxel(img_series)
    # print(slice_dose)
    # np.save("/Users/mr.chai/PycharmProjects/auto_seg/seg_task/data/dose/dose.npy", slice_dose)
    if 'exclude_pt_list' in cfg:
        pass
        # todo: exclude some patient

    # normalization setting
    if cfg.norm_method == 'fix':
        img_series.norm_method = 'fix'
        img_series.min_value = cfg.norm_low
        img_series.max_value = cfg.norm_high

    slice_img = img_series.normalized_voxel_array
    # roi_dicts = cfg.roi_dicts

    # slice_mask = np.zeros(list(slice_img.shape) + [len(roi_dicts)], dtype=int)
    # all_roi_exist = rtst_dataset.roi_exist_check_and_match(roi_dicts)

    # if (not all_roi_exist) and cfg.remove_data_with_problems:
    #     logger.warning('not generate h5py file for [{}].'.format(pt_id))
    #     return None

    # for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):
    #     roi_mask, success_flag = rtst_dataset.create_3d_mask(roi_ind_in_conf, img_series)
    #     slice_mask[:, :, :, roi_ind_in_conf] = roi_mask
    print("slice_img.shape:",slice_img.shape)
    # print("slice_mask.shape:",slice_mask.shape)
    # print("slice_dose.shape:", slice_dose.shape)
    print("slice_dose_max:", np.max(slice_dose))
    slice_img = shape_check_and_rescale(slice_img, cfg.dim_y, cfg.dim_x)
    # slice_mask = shape_check_and_rescale(slice_mask, cfg.dim_y, cfg.dim_x)
    slice_dose = shape_check_and_rescale(slice_dose, cfg.dim_y, cfg.dim_x)

    data_id_str = str(uuid.uuid4())[:8]
    # output_path = random_path(cfg)

    with h5py.File(output_path + os.sep + data_id_str + ".h5", "w") as hf:
        hf.create_dataset("slice_img", data=slice_img, compression='lzf')
        # hf.create_dataset("slice_mask", data=slice_mask, compression='lzf')
        hf.create_dataset("slice_dose", data=slice_dose, compression='lzf')

    with open(output_path + os.sep + "data_recoder.csv", 'a+') as csv_file:
        csv_file.write(data_id_str + ',' + pt_id + ',' +
                       str(img_series.z_dim) + ',' + '\n')
    logger.info('[{}] h5py file generate success.'.format(pt_id))


# === block 10. dose, rtst test data === start
def gen_dose_rtst_test_data(cfg):
    dicom_path = to_abs_path(cfg.predict_dicom_path)
    # train_path = to_abs_path(cfg.train_path)
    # validation_path = to_abs_path(cfg.validation_path)
    test_path = to_abs_path(cfg.test_out_path)
    dicomdir = DicomDirectory(dicom_path)
    dicomdir.scan()
    clean_folder(test_path)
    # clean_folder(validation_path)

    logger.info('start create training and validation hdf5 dataset.')
    if 'sort_method' in cfg:
        logger.info('sort method: ' + cfg.sort_method)
    else:
        logger.info('default sort method: instance number')

    if cfg.thread_num == 1:
        for dicom_dose_series, dicom_rtst_series, img_series_info in dicomdir.dose_rtst_and_its_img_series_iter():
            output_path = test_path
            _dose_rtst_test_data(dicom_dose_series, dicom_rtst_series, img_series_info, cfg, output_path)
    else:
        with multiprocessing.Pool(processes=cfg.thread_num) as pool:
            dose_list = [dose for dose, rtst, img in dicomdir.dose_rtst_and_its_img_series_iter()]
            rtst_list = [rtst for dose, rtst, img in dicomdir.dose_rtst_and_its_img_series_iter()]
            img_list = [img for dose, rtst, img in dicomdir.dose_rtst_and_its_img_series_iter()]
            output_path_list = [test_path for i in range(len(img_list))]
            pool.starmap(_dose_rtst_test_data,
                         zip(dose_list, rtst_list, img_list, repeat(cfg), output_path_list))


def _dose_rtst_test_data(dicom_dose_series, dicom_rtst_series, img_series_info, cfg, output_path):
    img_series = DicomCTSeries(img_series_info)
    if 'sort_method' in cfg:
        img_series.set_sort_method(cfg.sort_method)
    img_series.load_data()

    rtst_file = dicom_rtst_series.single_file
    rtst_dataset = DicomRTST(rtst_file)
    pt_id = rtst_dataset.PatientID

    dose_dataset = DicomDoseSeries(dicom_dose_series)
    dose_dataset.load_data()
    slice_dose = dose_dataset.to_img_voxel(img_series)
    # print(slice_dose)
    # np.save("/Users/mr.chai/PycharmProjects/auto_seg/seg_task/data/dose/dose.npy", slice_dose)
    if 'exclude_pt_list' in cfg:
        pass
        # todo: exclude some patient

    # normalization setting
    if cfg.norm_method == 'fix':
        img_series.norm_method = 'fix'
        img_series.min_value = cfg.norm_low
        img_series.max_value = cfg.norm_high

    slice_img = img_series.normalized_voxel_array
    roi_dicts = cfg.roi_dicts

    slice_mask = np.zeros(list(slice_img.shape) + [len(roi_dicts) + 1], dtype=int)
    all_roi_exist = rtst_dataset.roi_exist_check_and_match(roi_dicts)

    if (not all_roi_exist) and cfg.remove_data_with_problems:
        logger.warning('not generate h5py file for [{}].'.format(pt_id))
        return None

    for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):
        roi_mask, success_flag = rtst_dataset.create_3d_mask(roi_ind_in_conf, img_series)
        slice_mask[:, :, :, roi_ind_in_conf] = roi_mask

    bc = BeamContourGeneration(slice_mask)
    slice_beam = bc.output()
    slice_mask[:, :, :, -1] = slice_beam

    slice_img = shape_check_and_rescale(slice_img, cfg.dim_y, cfg.dim_x)
    slice_mask = shape_check_and_rescale(slice_mask, cfg.dim_y, cfg.dim_x)
    slice_dose = shape_check_and_rescale(slice_dose, cfg.dim_y, cfg.dim_x)

    data_id_str = str(uuid.uuid4())[:8]
    # output_path = random_path(cfg)

    with h5py.File(output_path + os.sep + pt_id + ".h5", "w") as hf:
        hf.create_dataset("slice_img", data=slice_img, compression='lzf')
        hf.create_dataset("slice_mask", data=slice_mask, compression='lzf')
        hf.create_dataset("slice_dose", data=slice_dose, compression='lzf')

    # with open(output_path + os.sep + "data_recoder.csv", 'a+') as csv_file:
    #     csv_file.write(data_id_str + ',' + pt_id + ',' +
    #                    str(img_series.z_dim) + ',' + '\n')
    logger.info('[{}] h5py file generate success.'.format(pt_id))