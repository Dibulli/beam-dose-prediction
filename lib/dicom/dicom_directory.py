# I hope I can use this for 2 years
# 2019.12.17 Jiazhou Wang
# Email: wjiazhou@gmail.com

import logging
import os

from lib.dicom.dicom_file import dcm_scan
from lib.dicom.dicom_files_base_class import DicomFilesBaseClass
from lib.dicom.dicom_patient import DicomPatient
from lib.dicom.dicom_frame import DicomFrame
from lib.utilities import get_all_files_in_the_folder


logger = logging.getLogger(__name__)


# only storage base information to save memory
class DicomDirectory(DicomFilesBaseClass):
    """
    patient_id -> DICOMPatient

    """

    def __init__(self, dicom_folder_path=None):
        super(DicomDirectory).__init__()

        # confirm the input path is str or list
        if isinstance(dicom_folder_path, str):
            # check input first
            if not os.path.isdir(dicom_folder_path):
                logger.error('the input [' + dicom_folder_path + '] is not a valid location.')
        elif isinstance(dicom_folder_path, list):
            for folder_i in dicom_folder_path:
                if not os.path.isdir(folder_i):
                    logger.error('the input [' + folder_i + '] is not a valid location.')
        else:
            logger.error('the input should be list or string.')

        self.dicom_folder_path = dicom_folder_path

        # base information
        self.pt_num = 0
        self.study_num = 0
        self.series_num = 0
        self.series_type_num = {'MR Image Storage': 0,
                                'CT Image Storage': 0,
                                'Positron Emission Tomography Image Storage': 0,
                                'RT Structure Set Storage': 0,
                                'RT Plan Storage': 0,
                                'RT Dose Storage': 0}
        self.instance_type_num = {'MR Image Storage': 0,
                                  'CT Image Storage': 0,
                                  'Positron Emission Tomography Image Storage': 0,
                                  'RT Structure Set Storage': 0,
                                  'RT Plan Storage': 0,
                                  'RT Dose Storage': 0}
        self.dicom_file_num = 0
        self.frame_dict = dict()

        self._series_list = None
        self._instance_list = None

    # key function of this class
    def scan(self):
        if isinstance(self.dicom_folder_path, str):
            folder_list = [self.dicom_folder_path]
        else:
            folder_list = self.dicom_folder_path

        for folder_i in folder_list:
            logger.info("start to scan folder [%s].", folder_i)
            file_list = get_all_files_in_the_folder(folder_i)
            # Loop all files in this directory
            for file_name in file_list:
                dicom_dataset = dcm_scan(file_name)
                if dicom_dataset is not None:
                    self.add_dicom_dataset(dicom_dataset)

        self.stat()
        self.brief_summary()

    def add_dicom_dataset(self, dicom_dataset):
        try:
            # get the same attributes of the DICOM files
            patient_id = dicom_dataset.PatientID
            frame_of_reference_uid = dicom_dataset.FrameOfReferenceUID
        except AttributeError:
            logger.warning('lack some key attributes!')
            logger.warning('file name {}'.format(dicom_dataset.filename))
            # do nothing
            return None

        # create new element
        if patient_id not in self.keys():
            self[patient_id] = DicomPatient(patient_id)

        if frame_of_reference_uid not in self.frame_dict.keys():
            self.frame_dict[frame_of_reference_uid] = DicomFrame(frame_of_reference_uid)

        self.frame_dict[frame_of_reference_uid].add_dicom_dataset(dicom_dataset)
        self[patient_id].add_dicom_dataset(dicom_dataset)

    def series_iter(self, series_type=None):
        for pt in self.values():
            for study in pt.values():
                for series in study.values():
                    if series_type is not None:
                        if series.sop_class_name == series_type:
                            yield series

                    else:
                        yield series

    def file_iter(self, series_type=None, patient_id=None):
        if patient_id is None:
            for pt in self.values():
                for study in pt.values():
                    for series in study.values():
                        if series_type is not None:
                            if series.sop_class_name == series_type:
                                for uid, file in series.items():
                                    yield uid, file
                        else:
                            for file in series.values():
                                yield file
        else:
            for pt in self.values():
                if pt.patient_id == patient_id:
                    for study in pt.values():
                        for series in study.values():
                            if series_type is not None:
                                if series.sop_class_name == series_type:
                                    for uid, file in series.items():
                                        yield uid, file
                            else:
                                for file in series.values():
                                    yield file

    def get_rtst_series_by_ref_img_uid(self, input_img_series_uid):
        for dicom_rtst_series in self.series_iter(series_type='RT Structure Set Storage'):
            ref_uid = dicom_rtst_series.referenced_series_uid
            if input_img_series_uid == ref_uid:
                return dicom_rtst_series
        return None

    def get_dose_series_by_frame_ref_uid(self, frame_ref_uid):
        for dicom_dose_series in self.series_iter(series_type='RT Dose Storage'):
            dose_frame_ref_uid = dicom_dose_series.frame_of_reference_uid
            if frame_ref_uid == dose_frame_ref_uid:
                return dicom_dose_series
        return None

    def multi_img_series_iter(self, series_info_list=None):
        if series_info_list is None:
            raise ValueError
        else:
            # set primary series
            primary_series = series_info_list[0]
            primary_use_fuzzy_match = primary_series['use_fuzzy_match']
            primary_series_description = primary_series['series_description']

            # loop all mr series
            for dicom_mr_series in self.series_iter(series_type='MR Image Storage'):
                patient_id = dicom_mr_series.patient_id

                # check primary series
                is_parimary_series = False
                if isinstance(primary_series_description, str):
                    if primary_use_fuzzy_match:
                        if primary_series_description in dicom_mr_series.SeriesDescription:
                            logger.info("[{}] primary series match  [{}]-->[{}]".format(
                                patient_id, primary_series_description, dicom_mr_series.SeriesDescription))
                            is_parimary_series = True
                    else:
                        if primary_series_description == dicom_mr_series.SeriesDescription:
                            logger.info("[{}] primary series match [{}]-->[{}]".format(
                                patient_id, primary_series_description, dicom_mr_series.SeriesDescription))
                            is_parimary_series = True
                else:
                    for series_description_i in primary_series_description:
                        if primary_use_fuzzy_match:
                            if series_description_i in dicom_mr_series.SeriesDescription:
                                logger.info("[{}] primary series match [{}]-->[{}]".format(
                                    patient_id, series_description_i, dicom_mr_series.SeriesDescription))
                                is_parimary_series = True
                                break
                        else:
                            if series_description_i == dicom_mr_series.SeriesDescription:
                                logger.info("[{}] primary series match [{}]-->[{}]".format(
                                    patient_id, series_description_i, dicom_mr_series.SeriesDescription))
                                is_parimary_series = True
                                break

                if is_parimary_series:

                    frame_of_reference_uid = dicom_mr_series.frame_of_reference_uid
                    frame_of_ref = self.frame_dict[frame_of_reference_uid]

                    # loop description lise
                    find_all_series_flag = True
                    img_series_info_list = []
                    for series_info in series_info_list:
                        find_series_flag = False

                        series_description = series_info['series_description']
                        use_fuzzy_match = series_info['use_fuzzy_match']

                        # loop series
                        for study in frame_of_ref.values():
                            for series in study.values():

                                # find this series
                                # todo: I have to use a lot if, which may need rewrite!!!
                                if isinstance(series_description, str):
                                    if use_fuzzy_match:
                                        if series_description in series.SeriesDescription:
                                            find_series_flag = True
                                            logger.info("[{}] series match  [{}]-->[{}]".format(
                                                patient_id, series_description, series.SeriesDescription))
                                            ref_uid = series.series_instance_uid
                                            img_series_info = self.get_series(ref_uid)
                                            img_series_info_list.append(img_series_info)

                                    else:
                                        if series_description == series.SeriesDescription:
                                            find_series_flag = True
                                            logger.info("[{}] series match [{}]-->[{}]".format(
                                                patient_id, series_description, series.SeriesDescription))
                                            ref_uid = series.series_instance_uid
                                            img_series_info = self.get_series(ref_uid)
                                            img_series_info_list.append(img_series_info)
                                else:
                                    for series_description_i in series_description:
                                        if use_fuzzy_match:
                                            if series_description_i in series.SeriesDescription:
                                                find_series_flag = True
                                                logger.info("[{}] series match [{}]-->[{}]".format(
                                                    patient_id, series_description_i, series.SeriesDescription))
                                                ref_uid = series.series_instance_uid
                                                img_series_info = self.get_series(ref_uid)
                                                img_series_info_list.append(img_series_info)
                                                break
                                        else:
                                            if series_description_i == series.SeriesDescription:
                                                find_series_flag = True
                                                logger.info("[{}] series match [{}]-->[{}]".format(
                                                    patient_id, series_description_i, series.SeriesDescription))
                                                ref_uid = series.series_instance_uid
                                                img_series_info = self.get_series(ref_uid)
                                                img_series_info_list.append(img_series_info)
                                                break

                                if find_series_flag:
                                    break
                            if find_series_flag:
                                break

                        if not find_series_flag:
                            logger.warning("[{}] not find matched series for [{}]".format(
                                patient_id, series_description))
                            logger.warning("The series in this patients.")
                            for study in frame_of_ref.values():
                                for series in study.values():
                                    logger.info(series.SeriesDescription)

                            find_all_series_flag = find_all_series_flag and find_series_flag
                    if find_all_series_flag:
                        yield img_series_info_list
                    else:
                        logger.warning("not process [{}] due to lack series".format(patient_id))
                else:
                    # not found the primary series
                    pass

    def dose_rtst_and_its_img_series_iter(self):
        for dicom_rtst_series in self.series_iter(series_type='RT Structure Set Storage'):
            ref_uid = dicom_rtst_series.referenced_series_uid
            img_series_info = self.get_series(ref_uid)
            img_frame_ref_uid = img_series_info.frame_of_reference_uid
            dose_series = self.get_dose_series_by_frame_ref_uid(img_frame_ref_uid)
            yield dose_series, dicom_rtst_series, img_series_info

    def rtst_and_its_img_series_iter(self, series_info_list=None):
        if series_info_list is None:
            for dicom_rtst_series in self.series_iter(series_type='RT Structure Set Storage'):
                ref_uid = dicom_rtst_series.referenced_series_uid
                img_series_info = self.get_series(ref_uid)
                yield dicom_rtst_series, img_series_info
        else:
            for dicom_rtst_series in self.series_iter(series_type='RT Structure Set Storage'):
                patient_id = dicom_rtst_series.patient_id

                frame_of_reference_uid = dicom_rtst_series.frame_of_reference_uid
                frame_of_ref = self.frame_dict[frame_of_reference_uid]

                # loop description lise
                find_all_series_flag = True
                img_series_info_list = []
                for series_info in series_info_list:
                    find_series_flag = False

                    series_description = series_info['series_description']
                    use_fuzzy_match = series_info['use_fuzzy_match']

                    # loop series
                    for study in frame_of_ref.values():
                        for series in study.values():

                            # find this series
                            # todo: I have to use a lot if, which may need rewrite!!!
                            if isinstance(series_description, str):
                                if use_fuzzy_match:
                                    if series_description in series.SeriesDescription:
                                        find_series_flag = True
                                        logger.debug("[{}] series match  [{}]-->[{}]".format(
                                            patient_id, series_description, series.SeriesDescription))
                                        ref_uid = series.series_instance_uid
                                        img_series_info = self.get_series(ref_uid)
                                        img_series_info_list.append(img_series_info)

                                else:
                                    if series_description == series.SeriesDescription:
                                        find_series_flag = True
                                        logger.debug("[{}] series match [{}]-->[{}]".format(
                                            patient_id, series_description, series.SeriesDescription))
                                        ref_uid = series.series_instance_uid
                                        img_series_info = self.get_series(ref_uid)
                                        img_series_info_list.append(img_series_info)
                            else:
                                for series_description_i in series_description:
                                    if use_fuzzy_match:
                                        if series_description_i in series.SeriesDescription:
                                            find_series_flag = True
                                            logger.debug("[{}] series match [{}]-->[{}]".format(
                                                patient_id, series_description_i, series.SeriesDescription))
                                            ref_uid = series.series_instance_uid
                                            img_series_info = self.get_series(ref_uid)
                                            img_series_info_list.append(img_series_info)
                                            break
                                    else:
                                        if series_description_i == series.SeriesDescription:
                                            find_series_flag = True
                                            logger.debug("[{}] series match [{}]-->[{}]".format(
                                                patient_id, series_description_i, series.SeriesDescription))
                                            ref_uid = series.series_instance_uid
                                            img_series_info = self.get_series(ref_uid)
                                            img_series_info_list.append(img_series_info)
                                            break

                            if find_series_flag:
                                break
                        if find_series_flag:
                            break

                    if not find_series_flag:
                        logger.warning("[{}] not find matched series for [{}]".format(
                            patient_id, series_description))
                        logger.warning("The series in this patients.")
                        for study in frame_of_ref.values():
                            for series in study.values():
                                logger.info(series.SeriesDescription)

                        find_all_series_flag = find_all_series_flag and find_series_flag
                if find_all_series_flag:
                    yield dicom_rtst_series, img_series_info_list
                else:
                    logger.warning("not process [{}] due to lack series".format(patient_id))

    def stat(self):
        for pt in self.values():
            self.pt_num = self.pt_num + 1
            for study in pt.values():
                self.study_num = self.study_num + 1
                for series in study.values():
                    self.series_num = self.series_num + 1
                    if series.sop_class_name in self.series_type_num.keys():
                        self.series_type_num[series.sop_class_name] = self.series_type_num[series.sop_class_name] + 1
                    if series.sop_class_name in self.instance_type_num.keys():
                        self.instance_type_num[series.sop_class_name] = self.instance_type_num[series.sop_class_name] + len(series)
                    self.dicom_file_num = self.dicom_file_num + len(series)

    def brief_summary(self):
        logger.info('find [{0:d}] patients.'.format(self.pt_num))
        logger.info('find [{0:d}] studies.'.format(self.study_num))
        logger.info('find [{0:d}] series.'.format(self.series_num))
        for series_type, num in self.series_type_num.items():
            logger.info('find [{0:d}] "{1:s}" series.'.format(num, series_type))
        for series_type, num in self.instance_type_num.items():
            logger.info('find [{0:d}] "{1:s}" instances.'.format(num, series_type))
        logger.info('find [{0:d}] dicom files.'.format(self.dicom_file_num))

    @property
    def series_list(self):
        if self._series_list is not None:
            return self._series_list
        else:
            self._series_list = {}
            for pt in self.values():
                for study in pt.values():
                    self._series_list.update(study)
            return self._series_list

    def get_series(self, series_uid):
        if series_uid in self.series_list.keys():
            return self.series_list[series_uid]
        else:
            logger.warning('not find this series [%s]', series_uid)
            return []

    @property
    def instance_list(self):
        if self._instance_list is not None:
            return self._instance_list
        else:
            self._instance_list = {}
            for pt in self.values():
                for study in pt.values():
                    for series in study.values():
                        self._instance_list.update(series)
            return self._instance_list

    def get_instance(self, instance_uid):
        if instance_uid in self.instance_list.keys():
            return self.instance_list[instance_uid]
        else:
            logger.warning('not find this instance [%s]', instance_uid)
            return []

if __name__ == '__main__':
    dicom_dir = DicomDirectory('../data/raw_dicom/')
    dicom_dir.scan()
    for series in dicom_dir.series_iter('RT Structure Set Storage'):
        ref_uid = series.referenced_series_uid
        img_series = dicom_dir.get_series(ref_uid)
