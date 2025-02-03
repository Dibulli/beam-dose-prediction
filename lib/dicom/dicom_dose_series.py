# I hope I can use this for 2 years
# 2020.4.29 Jiazhou Wang
# Email: wjiazhou@gmail.com

import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from lib.dicom.dicom_image_file import DicomImageFile
from lib.dicom.dicom_image_series import DicomImageSeries
from lib.dicom.dicom_series import DicomSeries

logger = logging.getLogger(__name__)


class DicomDoseSeries(DicomSeries):
    def __init__(self, *args, **kwargs):
        super(DicomDoseSeries, self).__init__(*args, **kwargs)

        self.voxel_array = None
        self._abs2vox_4d = None
        self._vox2abs_4d = None

    def load_data(self):
        print("this dose series has [{:}] sub-series".format(len(self.values())))
        file_name = list(self.values())[0]
        dicom_dataset = DicomImageFile(file_name)

        # loop all tages
        for tag_name in dicom_dataset.dir():
            try:
                dataset_attr = dicom_dataset.__getattr__(tag_name)
            except AttributeError:
                # I dont know why!!!
                logger.debug('tag "{}" is not in the file: "{}".'.format(
                    tag_name, file_name
                ))
                dataset_attr = None

            # initial dict
            self.__setattr__(tag_name, dataset_attr)

        # set voxel array
        self.voxel_array = dicom_dataset.pixel_array * dicom_dataset.DoseGridScaling

    def load_data_by_file_name(self, file_name):
        print("this dose series has [{:}] sub-series".format(len(self.values())))
        dicom_dataset = DicomImageFile(file_name)

        # loop all tages
        for tag_name in dicom_dataset.dir():
            try:
                dataset_attr = dicom_dataset.__getattr__(tag_name)
            except AttributeError:
                # I dont know why!!!
                logger.debug('tag "{}" is not in the file: "{}".'.format(
                    tag_name, file_name
                ))
                dataset_attr = None

            # initial dict
            self.__setattr__(tag_name, dataset_attr)

        # set voxel array
        self.voxel_array = dicom_dataset.pixel_array * dicom_dataset.DoseGridScaling

    @property
    def vox2abs_4d(self):
        # monaco dcm dose 和 pinnacle dcm dose 格式不同。dicom dose的层厚不是SliceThickness（这个是CT扫描的层厚，
        # 如果剂量计算网格和ct扫描层厚相同则不会报错。），而是GrdiFrameOffsetVector参数里的相邻两项之间相差的距离（mm）。

        # 同时ct序列排序可能也存在问题，InstanceNumber=0 可能不是最脚端的那层ct。目前monaco的数据，暂时先把dicom_img_series里的sort方案改为反向sort。

        if self._vox2abs_4d is None:
            delta_r = self.PixelSpacing[0]
            delta_c = self.PixelSpacing[1]
            # delta_z = self.SliceThickness
            delta_z = self.GridFrameOffsetVector[1] - self.GridFrameOffsetVector[0]

            T11 = self.ImagePositionPatient[0]
            T12 = self.ImagePositionPatient[1]
            T13 = self.ImagePositionPatient[2]
            image_orientation = self.ImageOrientationPatient
            F11 = image_orientation[0]
            F21 = image_orientation[1]
            F31 = image_orientation[2]
            F12 = image_orientation[3]
            F22 = image_orientation[4]
            F32 = image_orientation[5]

            F3 = np.cross([F11, F21, F31], [F12, F22, F32]).T
            F13 = F3[0]
            F23 = F3[1]
            F33 = F3[2]

            self._vox2abs_4d = np.array(
                [[F11 * delta_r, F12 * delta_c, F13 * delta_z, T11],
                 [F21 * delta_r, F22 * delta_c, F23 * delta_z, T12],
                 [F31 * delta_r, F32 * delta_c, F33 * delta_z, T13],
                 [0, 0, 0, 1]])
        return self._vox2abs_4d

    @property
    def abs2vox_4d(self):
        if self._abs2vox_4d is None:
            self._abs2vox_4d = np.linalg.inv(self.vox2abs_4d)
        return self._abs2vox_4d

    def to_img_voxel(self, dicom_img_series):
        assert isinstance(dicom_img_series, DicomImageSeries)

        # initial image coordinate dose matrix
        dose_mat_img_coord = np.zeros((dicom_img_series.Rows, dicom_img_series.Columns, dicom_img_series.z_dim),
                                      dtype=float)

        # get the affine matrix ct->abs
        ct2abs = dicom_img_series.series_vox2abs

        # get the affine matrix abs->dose
        abs2dose = self.abs2vox_4d

        # get the affine matrix ct->dose
        ct2dose = np.matmul(abs2dose, ct2abs)

        # create interpolation function
        z_coords = np.arange(self.NumberOfFrames)
        x_coords = np.arange(self.Rows)
        y_coords = np.arange(self.Columns)
        dose_interpolating_function = RegularGridInterpolator((z_coords, x_coords, y_coords),
                                                              self.voxel_array,
                                                              method='linear',
                                                              bounds_error=False,
                                                              fill_value=0)

        # interpolation
        # create the ct grid on dose coordinate
        # 问题铁定出在这里。z插值的很奇怪哦

        for z in np.arange(dicom_img_series.z_dim):

            xi, yi = np.meshgrid(np.arange(dicom_img_series.Rows), np.arange(dicom_img_series.Columns))
            zi = np.ones_like(xi) * z
            ii = np.ones_like(xi)
            xi = xi.flatten()
            yi = yi.flatten()
            zi = zi.flatten()
            ii = ii.flatten()

            ct_points = np.row_stack((xi, yi, zi, ii))
            dose_points = np.dot(ct2dose, ct_points)
            dose_slice = dose_interpolating_function(dose_points[[2, 1, 0], :].T)
            dose_slice = dose_slice.reshape(dicom_img_series.Rows, dicom_img_series.Columns)

            dose_mat_img_coord[:, :, dicom_img_series.z_dim - z - 1] = dose_slice
            logger.info("max dose after interpolator is: [{:.2f}]".format(np.max(dose_slice)))
            # print("max dose after interpolator is: [{:.2f}]".format(np.max(dose_slice)))

            # logger.info("max dose after interpolator is: [{:.2f}]".format(np.max(self.voxel_array[z, :, :])))
            logger.info("")

        logger.info('interpolation done!')
        return dose_mat_img_coord

if __name__ == "__main__":
    dose_file = DicomDoseSeries()
