# I hope I can use this for 2 years
# 2019.12.17 Jiazhou Wang
# Email: wjiazhou@gmail.com

import logging
import math
import os
import cv2
import numpy as np

from lib.dicom.dicom_image_file import DicomImageFile
from lib.dicom.dicom_series import DicomSeries

logger = logging.getLogger(__name__)


def image_sort_patient_position_and_z(dicom_image_series):
    if 'H' in dicom_image_series.patient_position:
        reverse_setting = False
    else:
        reverse_setting = True
    sort_list = list(uid for uid, _ in sorted(dicom_image_series.items(),
                                              key=lambda item: float(item[1].ImagePositionPatient[2]),
                                              reverse=reverse_setting))
    return sort_list


# z from small to large
def image_sort_z(dicom_image_series):
    sort_list = list(uid for uid, _ in sorted(dicom_image_series.items(),
                                              key=lambda item: float(item[1].ImagePositionPatient[2]),
                                              reverse=False))
    return sort_list


# I think this sort method is the best method
def image_sort_instance_number(dicom_image_series):
    sort_list = list(uid for uid, _ in sorted(dicom_image_series.items(),
                                              key=lambda item: int(item[1].InstanceNumber),
                                              reverse=False))
    return sort_list


def _voxel_normization_factory(np_array, min_value, max_value):
    value_range = max_value - min_value
    scaled_voxel_array = (np_array - min_value) / value_range
    scaled_voxel_array = np.clip(scaled_voxel_array, 0, 1)
    return scaled_voxel_array


def voxel_normalization_min_max(dicom_image_series):
    max_value = np.max(dicom_image_series.voxel_array)
    min_value = np.min(dicom_image_series.voxel_array)
    return _voxel_normization_factory(dicom_image_series.voxel_array, min_value, max_value)


def voxel_normalization_percent_99_min_max(dicom_image_series):
    max_value = np.percentile(dicom_image_series.voxel_array, 99)
    min_value = np.percentile(dicom_image_series.voxel_array, 1)
    return _voxel_normization_factory(dicom_image_series.voxel_array, min_value, max_value)


# normalize value between 0 ~ 1
def voxel_normalization_fixed_scale(dicom_image_series):
    max_value = dicom_image_series.max_value
    min_value = dicom_image_series.min_value
    return _voxel_normization_factory(dicom_image_series.voxel_array, min_value, max_value)


class DicomImageSeries(DicomSeries):
    def __init__(self, *args, **kwargs):
        super(DicomImageSeries, self).__init__(*args, **kwargs)

        # a dict used to storage dataset
        # sop_instance_uid -> dataset
        self.dataset_dict = None

        self.sort_method = image_sort_instance_number
        self._sort_list = None

        self.norm_method = None
        self.min_value = None
        self.max_value = None

        self._voxel_array_cache = None
        self._affine_mat_list = None

    def clear_cache(self):
        self._voxel_array_cache = None

    def set_sort_method(self, sort_fun):
        self.sort_method = sort_fun

    @property
    def sorted_uid_list(self):
        if self._sort_list is None:
            if len(self) == 0:
                self._sort_list = []
            else:
                self._sort_list = self.sort_method(self.dataset_dict)
        return self._sort_list

    @property
    def z_dim(self):
        return len(self)

    def load_data(self):
        if self.dataset_dict is None:
            self.dataset_dict = {}

        # some tag sets
        all_tag_name_set = set()  # storage all tags
        non_unique_value_tag_name_set = set()  # storage tags have different value
        pre_dataset = None  # tmp variable

        # loop all files
        for sop_instance_uid, file_name in self.items():
            dicom_dataset = DicomImageFile(file_name)
            self.dataset_dict[sop_instance_uid] = dicom_dataset

            # loop all tages
            for tag_name in dicom_dataset.dir():
                if tag_name == 'SeriesDescription':
                    continue

                all_tag_name_set.add(tag_name)

                try:
                    dataset_attr = dicom_dataset.__getattr__(tag_name)
                except AttributeError:
                    # I dont know why!!!
                    logger.debug('tag "{}" is not in the file: "{}".'.format(
                        tag_name, file_name
                    ))
                    dataset_attr = None

                # first time
                if tag_name not in self.__dict__:
                    # initial dict
                    self.__setattr__(tag_name, {sop_instance_uid: dataset_attr})
                else:
                    # add items
                    self.__getattribute__(tag_name).update({sop_instance_uid: dataset_attr})

                    # do not check tag already in non unique
                    if tag_name not in non_unique_value_tag_name_set:
                        # check the value is unique or not

                        # this tag not exist in previous dataset
                        try:
                            pre_dataset_attr = pre_dataset.__getattr__(tag_name)
                        except AttributeError:
                            dataset_attr = None

                        if dataset_attr != pre_dataset_attr:
                            non_unique_value_tag_name_set.add(tag_name)

            pre_dataset = dicom_dataset
        del pre_dataset

        unique_value_tag_name_set = all_tag_name_set - non_unique_value_tag_name_set
        for tag_name in unique_value_tag_name_set:
            dataset_attr = self.__getattribute__(tag_name).popitem()[1]
            self.__setattr__(tag_name, dataset_attr)

    @property
    def voxel_array(self):
        if self._voxel_array_cache is None:
            output_array = np.zeros((self.Rows, self.Columns, self.z_dim), dtype=float)
            for z_ind, uid in enumerate(self.sorted_uid_list):
                output_array[:, :, z_ind] = self.dataset_dict[uid].pixel_array
            self._voxel_array_cache = output_array
            return output_array
        else:
            return self._voxel_array_cache

    @property
    def normalized_voxel_array(self):
        if self.norm_method == 'fix':
            return voxel_normalization_fixed_scale(self)
        elif self.norm_method == 'percent_99_min':
            return voxel_normalization_percent_99_min_max(self)
        else:
            return voxel_normalization_min_max(self)

    def generate_glance_image(self, img_per_row=10, uid_list=None):
        logger.info('create glance image for patient [{}]'.format(self.patient_id))
        if uid_list is None:
            row_num = math.ceil(self.z_dim / img_per_row)
            glance_image = np.zeros((self.Rows * row_num, self.Columns * img_per_row), dtype=float)
            for img_i in range(self.z_dim):
                x_start = img_i % img_per_row
                y_start = math.floor(img_i / img_per_row)
                # please note. column first the row
                glance_image[y_start * self.Rows: y_start * self.Rows + self.Rows,
                x_start * self.Columns: x_start * self.Columns + self.Columns] = self.voxel_array[:, :, img_i]
            return glance_image
        else:
            row_num = math.ceil(len(uid_list) / img_per_row)
            glance_image = np.zeros((self.Rows * row_num, self.Columns * img_per_row), dtype=float)
            for img_i, uid in enumerate(uid_list):
                x_start = img_i % img_per_row
                y_start = math.floor(img_i / img_per_row)
                # please note. column first the row
                ind = self.sorted_uid_list.index(uid)
                glance_image[y_start * self.Rows: y_start * self.Rows + self.Rows,
                x_start * self.Columns: x_start * self.Columns + self.Columns] = self.voxel_array[:, :, ind]
            return glance_image

    def save_glance_image(self, output_path):
        glance_image = self.generate_glance_image()
        # scale to the range of the opencv gray image
        glance_image_norm = (glance_image - np.min(glance_image)) / (np.max(glance_image) - np.min(glance_image))
        glance_image_norm = glance_image_norm * 256
        glacne_image_name = output_path + os.sep + self.patient_id + '.png'
        cv2.imwrite(glacne_image_name, glance_image_norm)

    @property
    def series_vox2abs(self):
        # todo: may have some error
        # first_slice_uid = self.sorted_uid_list[0]
        first_slice_uid = self.sorted_uid_list[-1]

        # print("uid of the first slice of this CT series is: [{}]".format(first_slice_uid))
        vox2abs = self.dataset_dict[first_slice_uid].vox2abs_4d
        return vox2abs

    @property
    def series_abs2vox(self):
        pass


class DicomCTSeries(DicomImageSeries):
    def __init__(self, *args, **kwargs):
        super(DicomCTSeries, self).__init__(*args, **kwargs)

        self.min_value = -1024
        self.max_value = 4096

    @property
    def voxel_array(self):
        output_array = super(DicomCTSeries, self).voxel_array
        if hasattr(self, 'RescaleSlope') and hasattr(self, 'RescaleIntercept'):
            return output_array * self.RescaleSlope + self.RescaleIntercept
        else:
            return output_array

    def _glance_image(self, view_window):
        glance_image = self.generate_glance_image()
        logger.debug('using window [{}]'.format(view_window))

        if view_window == 'lung':
            ww = 2000
            wl = -600
            glance_min = wl - ww / 2
            glance_range = ww
        elif view_window == 'mediastinum':
            ww = 350
            wl = 50
            glance_min = wl - ww / 2
            glance_range = ww
        elif view_window == 'bone':
            ww = 1500
            wl = 350
            glance_min = wl - ww / 2
            glance_range = ww
        elif view_window == 'soft':
            ww = 500
            wl = 60
            glance_min = wl - ww / 2
            glance_range = ww
        else:
            glance_min = np.min(glance_image)
            glance_max = np.max(glance_image)
            glance_range = glance_max - glance_min

        scaled_glance_image = np.clip((glance_image - glance_min) / glance_range, 0, 1)
        scaled_glance_image = scaled_glance_image * 255

        return scaled_glance_image

    def _check_image(self, view_window):
        logger.info('using window [{}]'.format(view_window))

        if view_window == 'lung':
            ww = 2000
            wl = -600
            check_min = wl - ww / 2
            check_range = ww
        elif view_window == 'mediastinum':
            ww = 350
            wl = 50
            check_min = wl - ww / 2
            check_range = ww
        elif view_window == 'bone':
            ww = 1500
            wl = 350
            check_min = wl - ww / 2
            check_range = ww
        elif view_window == 'soft':
            ww = 500
            wl = 60
            check_min = wl - ww / 2
            check_range = ww
        else:
            check_min = np.min(self.voxel_array)
            glance_max = np.max(self.voxel_array)
            check_range = glance_max - check_min

        scaled_check_image = np.clip((self.voxel_array - check_min) / check_range, 0, 1)
        scaled_check_image = scaled_check_image * 255

        return scaled_check_image

    @property
    def glance_image_lung(self):
        return self._glance_image('lung')

    @property
    def glance_image_soft(self):
        return self._glance_image('soft')

    @property
    def check_image_soft(self):
        return self._check_image('soft')

    @property
    def check_image_lung(self):
        return self._check_image('lung')

    def save_glance_image(self, output_path, view_window=None):
        scaled_glance_image = self._glance_image(view_window)
        glacne_image_name = output_path + os.sep + self.patient_id + '.png'
        cv2.imwrite(glacne_image_name, scaled_glance_image)


class DicomMRSeries(DicomImageSeries):
    def __init__(self, *args, **kwargs):
        super(DicomMRSeries, self).__init__(*args, **kwargs)

    @property
    def voxel_array(self):
        output_array = super(DicomMRSeries, self).voxel_array
        if hasattr(self, 'RescaleSlope') and hasattr(self, 'RescaleIntercept'):
            return output_array * self.RescaleSlope + self.RescaleIntercept
        else:
            return output_array

    @property
    def check_image(self):
        check_max = np.percentile(self.voxel_array, 99)
        check_min = np.percentile(self.voxel_array, 1)
        check_range = check_max - check_min
        scaled_check_image = np.clip((self.voxel_array - check_min) / check_range, 0, 1)
        scaled_check_image = scaled_check_image * 255
        return scaled_check_image

    @property
    def glance_image(self):
        glance_image = self.generate_glance_image()
        glance_max = np.percentile(glance_image, 99)
        glance_min = np.percentile(glance_image, 1)
        glance_range = glance_max - glance_min
        scaled_glance_image = np.clip((glance_image - glance_min) / glance_range, 0, 1)
        scaled_glance_image = scaled_glance_image * 255
        return scaled_glance_image

    def gen_glance_image_with_uid_list(self, selected_slice_index):
        glance_image = self.generate_glance_image(uid_list=selected_slice_index)
        glance_max = np.percentile(glance_image, 99)
        glance_min = np.percentile(glance_image, 1)
        glance_range = glance_max - glance_min
        scaled_glance_image = np.clip((glance_image - glance_min) / glance_range, 0, 1)
        scaled_glance_image = scaled_glance_image * 255
        return scaled_glance_image

    def save_glance_image(self, output_path, view_window=None):
        scaled_glance_image = self.glance_image(view_window)
        glacne_image_name = output_path + os.sep + self.patient_id + '.png'
        cv2.imwrite(glacne_image_name, scaled_glance_image)