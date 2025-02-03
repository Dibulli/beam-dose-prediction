# I hope I can use this for 2 years
# 2019.12.17 Jiazhou Wang
# Email: wjiazhou@gmail.com

import logging

import numpy as np

from lib.dicom.dicom_file import DicomFile

logger = logging.getLogger(__name__)


class DicomImageFile(DicomFile):
    def __init__(self, filename):
        super(DicomImageFile, self).__init__(filename)

        self._vox2abs = None
        self._abs2vox = None
        self._vox2abs_4d = None
        self._abs2vox_4d = None

    @property
    def vox2abs(self):
        if self._vox2abs is None:
            delta_r = self.PixelSpacing[0]
            delta_c = self.PixelSpacing[1]
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

            self._vox2abs = np.array(
                [[F11 * delta_r, F12 * delta_c, T11],
                 [F21 * delta_r, F22 * delta_c, T12],
                 [F31 * delta_r, F32 * delta_c, T13]])
            if np.linalg.matrix_rank(self.vox2abs) < 3:
                print(self._vox2abs)
                print("F11=image_orientation[0]:", F11)
                print("F12=image_orientation[3]:", F12)
                print("F21=image_orientation[1]:", F21)
                print("F22=image_orientation[4]:", F22)
                print("F31=image_orientation[2]:", F31)
                print("F32=image_orientation[5]:", F32)
                print("Delta_r=PixelSpacing[0]:", delta_r)
                print("Delta_c=PixelSpacing[1]:", delta_c)

                logger.warning('patient ID: ' + self.PatientID)
                logger.warning('This affine matrix rank is less than 3!')
                self.vox2abs[2, 2] = 1

        return self._vox2abs

    @property
    def vox2abs_4d(self):
        if self._vox2abs_4d is None:
            delta_r = self.PixelSpacing[0]
            delta_c = self.PixelSpacing[1]
            delta_z = self.SliceThickness
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
    def abs2vox(self):
        if self._abs2vox is None:
            self._abs2vox = np.linalg.inv(self.vox2abs)
        return self._abs2vox

    @property
    def abs2vox_4d(self):
        if self._abs2vox_4d is None:
            self._abs2vox_4d = np.linalg.inv(self.vox2abs_4d)
        return self._abs2vox_4d

    # contour_abs_list shape = [N, 3], (col, row, z)
    # return [N, 3]
    def points_abs2rel(self, contour_abs_points):
        contour_rel_points = np.transpose(np.dot(self.abs2vox, np.transpose(contour_abs_points)))
        return contour_rel_points

    # contour_rel_list [N, 3], (col, row, z)
    # return [N, 3]
    def points_rel2abs(self, contour_rel_points):
        contour_abs_points = np.transpose(np.dot(self.vox2abs, np.transpose(contour_rel_points)))
        return contour_abs_points

