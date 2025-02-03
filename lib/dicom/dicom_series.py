# I hope I can use this for 2 years
# 2019.12.17 Jiazhou Wang
# Email: wjiazhou@gmail.com

import logging

from pydicom.uid import UID

from .dicom_files_base_class import DicomFilesBaseClass

logger = logging.getLogger(__name__)


def get_rt_referenced_series_uid(dicom_dataset):
    series_instance_uid_list = list()
    referenced_frame_sequence = dicom_dataset.ReferencedFrameOfReferenceSequence
    for referenced_frame in referenced_frame_sequence:
        for rt_referenced_study in referenced_frame.RTReferencedStudySequence:
            for rt_referenced_series in rt_referenced_study.RTReferencedSeriesSequence:
                series_instance_uid_list.append(rt_referenced_series.SeriesInstanceUID)
    if len(series_instance_uid_list) == 1:
        return series_instance_uid_list[0]
    else:
        logger.warning('multi seriese was referenced!')
        return series_instance_uid_list


class DicomSeries(DicomFilesBaseClass):
    """
    dict: file_sop -> filename
    """

    def __init__(self, *args, **kwargs):

        self.patient_id = None
        self.series_instance_uid = None
        self.sop_class_uid = None
        self.sop_class_name = None
        self.frame_of_reference_uid = None
        self.SeriesDescription = None

        self.referenced_series_uid = None  # for rtst

        if len(args) == 1 and isinstance(args[0], str):
            self.series_instance_uid = args[0]
        else:
            super(DicomSeries, self).__init__(*args, **kwargs)

    def add_dicom_dataset(self, dicom_dataset):

        try:
            # get the same attributes of the DICOM files
            patient_id = dicom_dataset.PatientID
            sop_instance_uid = dicom_dataset.SOPInstanceUID
            series_instance_uid = dicom_dataset.SeriesInstanceUID
            frame_of_reference_uid = dicom_dataset.FrameOfReferenceUID
            sop_class_uid = dicom_dataset.SOPClassUID
        except AttributeError:
            logger.warning('lack some key attributes!')
            logger.warning('file name {}'.format(dicom_dataset.filename))
            # do nothing
            return None

        # do not add same file twice
        if sop_instance_uid in self.keys():
            logger.info("two files have same UID.")
            logger.info("file 1: [%s].", dicom_dataset.filename)
            logger.info("file 2: [%s].", self[sop_instance_uid])
            return None

        # read information from first file
        if self.series_instance_uid is None:
            self.series_instance_uid = series_instance_uid
        elif self.series_instance_uid != series_instance_uid:
            logger.warning('this file is not belong to this series!')
            logger.warning('file name {}'.format(dicom_dataset.filename))
            return None

        # set information and consistence check
        if self.sop_class_uid is None:
            self.patient_id = patient_id
            self.sop_class_uid = sop_class_uid
            self.sop_class_name = UID(self.sop_class_uid).name
            self.frame_of_reference_uid = frame_of_reference_uid
        elif self.sop_class_uid != sop_class_uid or self.frame_of_reference_uid != frame_of_reference_uid:
            logger.warning('this file is not consistence with other file in this series!')
            logger.warning('file name {}'.format(dicom_dataset.filename))
            return None

        if self.sop_class_name == 'RT Structure Set Storage':
            self.referenced_series_uid = get_rt_referenced_series_uid(dicom_dataset)

        self[sop_instance_uid] = dicom_dataset.filename
        if 'SeriesDescription' in dicom_dataset:
            self.SeriesDescription = dicom_dataset.SeriesDescription
        else:
            self.SeriesDescription = ''

    @property
    def single_file(self):
        if len(self) == 1:
            return list(self.values())[0]
        else:
            logger.warning('this series have more than 1 file!')