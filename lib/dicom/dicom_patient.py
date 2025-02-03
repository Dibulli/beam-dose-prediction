# I hope I can use this for 2 years
# 2019.12.17 Jiazhou Wang
# Email: wjiazhou@gmail.com

import logging
from .dicom_files_base_class import DicomFilesBaseClass
from .dicom_study import DicomStudy

logger = logging.getLogger(__name__)


class DicomPatient(DicomFilesBaseClass):
    """
    study_uid -> DicomStudy
    """
    def __init__(self, *args, **kwargs):
        self.patient_id = None

        if len(args) == 1 and isinstance(args[0], str):
            self.patient_id = args[0]
        else:
            super(DicomPatient, self).__init__(*args, **kwargs)

    def add_dicom_dataset(self, dicom_dataset):
        try:
            # get the same attributes of the DICOM files
            study_instance_uid = dicom_dataset.StudyInstanceUID
            patient_id = dicom_dataset.PatientID
        except AttributeError:
            logger.warning('lack some key attributes!')
            logger.warning('file name {}'.format(dicom_dataset.filename))
            # do nothing
            return None

        # read information from first file
        if self.patient_id is None:
            self.patient_id = patient_id
        elif self.patient_id != patient_id:
            logger.warning('this file is not belong to this patient!')
            logger.warning('file name {}'.format(dicom_dataset.filename))

        # create new element
        if study_instance_uid not in self.keys():
            self[study_instance_uid] = DicomStudy(study_instance_uid)

        self[study_instance_uid].add_dicom_dataset(dicom_dataset)