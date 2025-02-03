# I hope I can use this for 2 years
# 2019.12.17 Jiazhou Wang
# Email: wjiazhou@gmail.com

import datetime
import logging
import math
import os

import cv2
from skimage import measure
import numpy as np
import pydicom.uid as uid
from pydicom import Dataset
from pydicom import Sequence

from lib.dicom.dicom_file import DicomFile
from lib.dicom.dicom_image_series import DicomImageSeries
from lib.utilities import shape_check_and_rescale

logger = logging.getLogger(__name__)


class DicomRTST(DicomFile):
    def __init__(self, filename):

        super(DicomRTST, self).__init__(filename)

        # roi index is the position of roi in the roi sequence
        self.roi_name_to_roi_index = {}
        self.rtst_internal_index_reading()

        # parameter for mask create
        self.matched_roi_index = []

        self._cached_roi_ind = None
        self._cached_mask = None

    def clear_cache(self):
        self._cached_roi_ind = None
        self._cached_mask = None

    def roi_exist_check_and_match(self, roi_dicts):
        all_roi_exist = True

        # loop all roi in config
        for single_roi_conf in roi_dicts:
            single_roi_exist = False
            fuzzy = single_roi_conf['use_fuzzy_match']

            # change str to list
            if isinstance(single_roi_conf['name'], str):
                cur_roi_name_list = [single_roi_conf['name']]
            else:
                cur_roi_name_list = single_roi_conf['name']

            # loop all names for one roi config
            for conf_roi_name in cur_roi_name_list:
                ind_in_rtst_file = self.get_rtst_roi_index(conf_roi_name, use_fuzzy_match=fuzzy)

                # exist
                if ind_in_rtst_file is not None:
                    single_roi_exist = True
                    self.matched_roi_index.append(ind_in_rtst_file)
                    break

            # not exist
            if not single_roi_exist:
                all_roi_exist = False
                self.matched_roi_index.append(None)
                logger.warning('not find one roi in patient [{}].'.format(self.PatientID))
                logger.warning('roi name:' + str(cur_roi_name_list))

        return all_roi_exist

    def get_rtst_roi_index(self, roi_name, use_fuzzy_match=True):
        if use_fuzzy_match:
            for rtst_roi in self.roi_name_to_roi_index.keys():
                if roi_name.lower() == rtst_roi.lower():
                    logger.debug(
                        'using fuzzy matching for [{}], [{}]~[{}].'.format(self.PatientID, rtst_roi, roi_name))
                    return self.roi_name_to_roi_index[rtst_roi]
            return None
        else:
            if roi_name in self.roi_name_to_roi_index.keys():
                return self.roi_name_to_roi_index[roi_name]
            else:
                return None

    def get_roi_color(self, roi_name):
        roi_index = self.get_roi_index(roi_name)
        if roi_index is None:
            # use red contour
            roi_display_color = [255, 0, 0]
        else:
            roi_display_color = self.ROIContourSequence[roi_index].ROIDisplayColor
        return np.array(roi_display_color, dtype=int)

    def create_3d_mask(self, roi_ind_in_conf, dicom_img_series):
        assert isinstance(dicom_img_series, DicomImageSeries)

        # do not calculate twice
        if roi_ind_in_conf == self._cached_roi_ind:
            return self._cached_mask

        # initial the matrix
        mask = np.zeros((dicom_img_series.Rows, dicom_img_series.Columns, dicom_img_series.z_dim), dtype=np.uint8)

        # find this roi
        roi_ind_in_rtst = self.matched_roi_index[roi_ind_in_conf]

        if roi_ind_in_rtst is None:
            return mask, False

        contour_sequence = self.ROIContourSequence[roi_ind_in_rtst].ContourSequence

        for contour in contour_sequence:
            contour_data = contour.ContourData

            if len(contour_data) == 0:
                continue

            # assume the points set only have one reference image
            if len(contour.ContourImageSequence) != 1:
                logger.warning('multi-referenced images, patient [{}] , file name [{}]'.
                               format(self.PatientID, self.filename))
                continue

            referenced_sop_instance_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            # get the location of this contour
            if referenced_sop_instance_uid in dicom_img_series.sorted_uid_list:
                mask_ind = dicom_img_series.sorted_uid_list.index(referenced_sop_instance_uid)
            else:
                logger.warning('referenced image not in the series, patient [{}] , file name [{}]'.
                               format(self.PatientID, self.filename))
                # TODO: use tranfer location to decide the mask on image
                # TODO: seems no way to deal with this
                # This error seems is lack image, maybe image is dropped when transfer
                # here I will remove this part of contour
                continue

            # change contour from absolute coordinate to relative coordinate
            contour_data_np = np.array(contour_data).reshape((-1, 3))
            if np.all(contour_data_np[:, 2]) == 0:
                contour_data_np[:, 2] = 1
            # 需要判断z轴绝对坐标是否为0。否则creat 3d mask 会出bug。
            rel_contour_points = dicom_img_series.dataset_dict[referenced_sop_instance_uid].points_abs2rel(
                contour_data_np)

            # opencv only accept int type data for drawContours
            rel_contour_points = np.round(rel_contour_points[:, 0:2]).astype(int)

            # process drawContours
            # 垃圾啊！！！！！！！！！！
            # 必须为 uint8
            grid = np.zeros((dicom_img_series.Rows, dicom_img_series.Columns), dtype=np.uint8)
            cv2.drawContours(image=grid, contours=[rel_contour_points],
                             contourIdx=-1, color=1, thickness=cv2.FILLED)

            if np.amax(grid) != 1:
                continue

            # update slice mask to volume mask
            if np.amax(mask[:, :, mask_ind]) == 0:
                mask[:, :, mask_ind] = grid
            else:
                # user XOR to combine mask
                mask[:, :, mask_ind] = mask[:, :, mask_ind] ^ grid

        self._cached_roi_ind = roi_ind_in_conf
        self._cached_mask = mask
        return mask, True

    def rtst_internal_index_reading(self):
        # Locate the name and number of each ROI
        if 'ROIContourSequence' in self:
            for roi_index, _ in enumerate(self.ROIContourSequence):
                StructureSetROI = self.StructureSetROISequence[roi_index]
                logger.debug("found ROI {}: {}".format(str(roi_index), StructureSetROI.ROIName))
                # todo: fix name list error
                self.roi_name_to_roi_index[StructureSetROI.ROIName] = roi_index

    def generate_check_roi_mask(self, roi_ind_in_conf, dicom_img_series):
        logger.debug('create roi checking image for patient [{}] roi index [{}]'.format(
            self.PatientID, roi_ind_in_conf))
        # first create 3d mask
        mask, _ = self.create_3d_mask(roi_ind_in_conf, dicom_img_series)
        return mask

    def generate_glance_roi_mask(self, roi_ind_in_conf, dicom_img_series, img_per_row=10):
        logger.debug('create roi glance image for patient [{}] roi index [{}]'.format(
            self.PatientID, roi_ind_in_conf))
        # first create 3d mask
        mask, _ = self.create_3d_mask(roi_ind_in_conf, dicom_img_series)

        # get the shape of the mask
        y_dim = mask.shape[0]
        x_dim = mask.shape[1]
        z_dim = mask.shape[2]

        row_num = math.ceil(z_dim / img_per_row)
        glance_roi_mask = np.zeros((y_dim * row_num, x_dim * img_per_row), dtype=float)

        for img_i in range(z_dim):
            x_start = img_i % img_per_row
            y_start = math.floor(img_i / img_per_row)
            # please note. column first the row
            glance_roi_mask[y_start * y_dim: y_start * y_dim + y_dim,
            x_start * x_dim: x_start * x_dim + x_dim] = mask[:, :, img_i]

        return glance_roi_mask


def masks_to_dicomrt(dicom_image_serires, roi_dicts, masks, roi_color_set, output_folder):
    logger.info('patient [{}] masks to dicom RTst!'.format(dicom_image_serires.PatientID))

    assert isinstance(dicom_image_serires, DicomImageSeries)
    dicom_rtst_dataset = Dataset()

    dicom_date = datetime.datetime.now().strftime('%Y%m%d')
    dicom_time = datetime.datetime.now().strftime('%H%M%S.%f')[0:10]
    rtst_series_instance_uid = uid.generate_uid()
    ref_frame_uid = dicom_image_serires.FrameOfReferenceUID

    # set file meta
    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    file_meta.MediaStorageSOPInstanceUID = rtst_series_instance_uid
    file_meta.ImplementationClassUID = uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
    dicom_rtst_dataset.file_meta = file_meta
    dicom_rtst_dataset.fix_meta_info()

    if hasattr(dicom_image_serires, 'SpecificCharacterSet'):
        dicom_rtst_dataset.SpecificCharacterSet = dicom_image_serires.SpecificCharacterSet
    else:
        dicom_rtst_dataset.SpecificCharacterSet = 'ISO_IR 192'
    dicom_rtst_dataset.InstanceCreationDate = dicom_date
    dicom_rtst_dataset.InstanceCreationTime = dicom_time
    dicom_rtst_dataset.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    dicom_rtst_dataset.SOPInstanceUID = rtst_series_instance_uid
    dicom_rtst_dataset.StudyDate = dicom_image_serires.StudyDate
    dicom_rtst_dataset.SeriesDate = dicom_date
    dicom_rtst_dataset.StudyTime = dicom_image_serires.StudyTime
    dicom_rtst_dataset.SeriesTime = dicom_time
    if hasattr(dicom_image_serires, 'AccessionNumber'):
        dicom_rtst_dataset.AccessionNumber = dicom_image_serires.AccessionNumber
    dicom_rtst_dataset.Modality = 'RTSTRUCT'
    dicom_rtst_dataset.Manufacturer = dicom_image_serires.Manufacturer
    if hasattr(dicom_image_serires, 'InstitutionName'):
        dicom_rtst_dataset.InstitutionName = dicom_image_serires.InstitutionName
    dicom_rtst_dataset.ReferringPhysicianName = ''
    dicom_rtst_dataset.StationName = ''
    dicom_rtst_dataset.StudyDescription = ''
    dicom_rtst_dataset.SeriesDescription = ''
    dicom_rtst_dataset.OperatorsName = ''
    dicom_rtst_dataset.ManufacturerModelName = ''
    dicom_rtst_dataset.PatientName = dicom_image_serires.PatientName
    dicom_rtst_dataset.PatientID = dicom_image_serires.PatientID
    dicom_rtst_dataset.PatientBirthDate = dicom_image_serires.PatientBirthDate
    dicom_rtst_dataset.PatientSex = dicom_image_serires.PatientSex
    if hasattr(dicom_image_serires, 'PatientAge'):
        dicom_rtst_dataset.PatientAge = dicom_image_serires.PatientAge
    else:
        dicom_rtst_dataset.PatientAge = '50Y'
    dicom_rtst_dataset.StudyInstanceUID = dicom_image_serires.StudyInstanceUID
    dicom_rtst_dataset.SeriesInstanceUID = rtst_series_instance_uid
    if hasattr(dicom_image_serires, 'StudyID'):
        dicom_rtst_dataset.StudyID = dicom_image_serires.StudyID
    else:
        dicom_rtst_dataset.StudyID = ''
    if hasattr(dicom_image_serires, 'SeriesNumber'):
        dicom_rtst_dataset.SeriesNumber = dicom_image_serires.SeriesNumber
    else:
        dicom_rtst_dataset.SeriesNumber = 1
    dicom_rtst_dataset.StructureSetLabel = ''
    dicom_rtst_dataset.StructureSetName = ''
    dicom_rtst_dataset.StructureSetDate = dicom_date
    dicom_rtst_dataset.StructureSetTime = dicom_time

    # ======== ReferencedFrameOfReferenceSequence ======== start
    # from small to big
    # contour
    contour_img_seq = Sequence()
    for img_sop_uid in dicom_image_serires.sorted_uid_list:
        contour_img = Dataset()
        contour_img.ReferencedSOPClassUID = dicom_image_serires.SOPClassUID
        contour_img.ReferencedSOPInstanceUID = img_sop_uid
        contour_img_seq.append(contour_img)

    # series
    rt_ref_series = Dataset()
    rt_ref_series.SeriesInstanceUID = dicom_image_serires.SeriesInstanceUID
    rt_ref_series.ContourImageSequence = contour_img_seq
    rt_ref_series_seq = Sequence()
    rt_ref_series_seq.append(rt_ref_series)

    # study
    rt_ref_study = Dataset()
    rt_ref_study.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'
    rt_ref_study.ReferencedSOPInstanceUID = dicom_image_serires.StudyInstanceUID
    rt_ref_study.RTReferencedSeriesSequence = rt_ref_series_seq
    rt_ref_study_seq = Sequence()
    rt_ref_study_seq.append(rt_ref_study)

    # frame
    ref_frame = Dataset()
    ref_frame.FrameOfReferenceUID = ref_frame_uid
    ref_frame.RTReferencedStudySequence = rt_ref_study_seq
    ref_frame_seq = Sequence()
    ref_frame_seq.append(ref_frame)

    # add to main dicom tree
    dicom_rtst_dataset.ReferencedFrameOfReferenceSequence = ref_frame_seq
    # ======== ReferencedFrameOfReferenceSequence ======== end

    # ======== StructureSetROISequence ======== start
    dicom_rtst_dataset.StructureSetROISequence = Sequence()
    for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):
        structure_set_roi = Dataset()
        structure_set_roi.ROINumber = roi_ind_in_conf + 1
        structure_set_roi.ReferencedFrameOfReferenceUID = ref_frame_uid
        structure_set_roi.ROIName = single_roi_conf.name
        structure_set_roi.ROIDescription = ''
        structure_set_roi.ROIGenerationAlgorithm = ''
        dicom_rtst_dataset.StructureSetROISequence.append(structure_set_roi)
    # ======== StructureSetROISequence ======== end

    # ======== ROIContourSequence ======== start
    dicom_rtst_dataset.ROIContourSequence = Sequence()
    for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):
        roi_contour = Dataset()
        color = roi_color_set[roi_ind_in_conf]
        roi_contour.ROIDisplayColor = color
        roi_contour.ContourSequence = Sequence()

        # masks [roi_c, col, row, z]
        mask = masks[single_roi_conf.mask_channel, :, :, :]
        mask = shape_check_and_rescale(mask,
                                       des_dim_y=dicom_image_serires.Columns,
                                       des_dim_x=dicom_image_serires.Rows)

        for z in range(mask.shape[2]):
            ref_img_uid = dicom_image_serires.sorted_uid_list[z]
            contours = measure.find_contours(np.pad(mask[:, :, z], 1, 'constant', constant_values=0),
                                             level=0.5,
                                             fully_connected='low',
                                             positive_orientation='high')

            for single_contour in contours:

                contour_bsplines = measure.subdivide_polygon(single_contour)
                contour_bsplines = measure.approximate_polygon(contour_bsplines, tolerance=0.5)
                contour_dataset = Dataset()

                contour_img = Dataset()
                contour_img.ReferencedSOPClassUID = dicom_image_serires.SOPClassUID
                contour_img.ReferencedSOPInstanceUID = ref_img_uid
                contour_img_seq = Sequence()
                contour_img_seq.append(contour_img)

                contour_dataset.ContourImageSequence = contour_img_seq
                contour_dataset.ContourGeometricType = 'CLOSED_PLANAR'

                # affine transform
                contour_data_np = contour_bsplines
                contour_data_np_extented = np.ones((contour_data_np.shape[0], contour_data_np.shape[1] + 1))
                contour_data_np_extented[:, 0] = contour_data_np[:, 1]
                contour_data_np_extented[:, 1] = contour_data_np[:, 0]
                abs_contour_points = dicom_image_serires.dataset_dict[ref_img_uid].points_rel2abs(
                    contour_data_np_extented)

                contour_data = list()
                for i in range(abs_contour_points.shape[0]):
                    point = abs_contour_points[i, :]
                    contour_data.append('{:.6f}'.format(float(point[0] - 1)))
                    contour_data.append('{:.6f}'.format(float(point[1] - 1)))
                    contour_data.append('{:.6f}'.format(float(point[2])))
                    # contour_data.append(dicom_image_serires.ImagePositionPatient[ref_img_uid][2])

                contour_dataset.NumberOfContourPoints = abs_contour_points.shape[0]
                contour_dataset.ContourData = contour_data
                roi_contour.ContourSequence.append(contour_dataset)

            roi_contour.ReferencedROINumber = roi_ind_in_conf + 1

        logger.info('[{}] tranform finished.'.format(single_roi_conf.name))
        dicom_rtst_dataset.ROIContourSequence.append(roi_contour)
    # ======== ROIContourSequence ======== end

    # ======== RTROIObservationsSequence ======== start
    dicom_rtst_dataset.RTROIObservationsSequence = Sequence()
    for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):
        roi_obs = Dataset()
        roi_obs.ObservationNumber = roi_ind_in_conf + 1
        roi_obs.ReferencedROINumber = roi_ind_in_conf + 1
        roi_obs.RTROIInterpretedType = ''
        roi_obs.ROIInterpreter = ''
        dicom_rtst_dataset.RTROIObservationsSequence.append(roi_obs)
    # ======== RTROIObservationsSequence ======== end

    # Set the transfer syntax
    dicom_rtst_dataset.is_little_endian = True
    dicom_rtst_dataset.is_implicit_VR = True
    rtst_filename = output_folder + os.sep + rtst_series_instance_uid + '.dcm'
    dicom_rtst_dataset.save_as(rtst_filename, write_like_original=False)
    logger.info('{} file saved.'.format(rtst_filename))
