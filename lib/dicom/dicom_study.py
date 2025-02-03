# I hope I can use this for 2 years
# 2019.12.17 Jiazhou Wang
# Email: wjiazhou@gmail.com

import logging

from .dicom_files_base_class import DicomFilesBaseClass
from .dicom_series import DicomSeries

logger = logging.getLogger(__name__)


class DicomStudy(DicomFilesBaseClass):
    """
    series_uid -> DicomSeries
    """
    def __init__(self, *args, **kwargs):
        self.study_instance_uid = None
        if len(args) == 1:
            self.study_instance_uid = args[0]
        else:
            super(DicomStudy).__init__(*args, **kwargs)

    # overwrite the method of add item
    def __setitem__(self, series_uid, dicom_series):
        # directly add

        # check items
        assert isinstance(dicom_series, DicomSeries)
        assert series_uid == dicom_series.series_instance_uid

        # add to dict
        dict.__setitem__(self, series_uid, dicom_series)

    def add_dicom_dataset(self, dicom_dataset):
        try:
            # get the same attributes of the DICOM files
            series_instance_uid = dicom_dataset.SeriesInstanceUID
            study_instance_uid = dicom_dataset.StudyInstanceUID
        except AttributeError:
            logger.warning('lack some key attributes!')
            logger.warning('file name {}'.format(dicom_dataset.filename))
            # do nothing
            return None

        # consistence check
        if self.study_instance_uid != study_instance_uid:
            logger.warning('this file is not belong to this study!')
            logger.warning('file name {}'.format(dicom_dataset.filename))
            return None

        # create new element
        if series_instance_uid not in self.keys():
            self[series_instance_uid] = DicomSeries(series_instance_uid)

        self[series_instance_uid].add_dicom_dataset(dicom_dataset)