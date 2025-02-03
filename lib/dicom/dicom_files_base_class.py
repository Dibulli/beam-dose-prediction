# I hope I can use this for 2 years
# 2019.12.17 Jiazhou Wang
# Email: wjiazhou@gmail.com

import logging
from abc import abstractmethod
from copy import deepcopy

logger = logging.getLogger(__name__)


class DicomFilesBaseClass(dict):
    def __init__(self, *args, **kwargs):

        if len(args) == 1:
            if isinstance(args[0], DicomFilesBaseClass):
                dicom_files = args[0]
                # copy dict
                super(DicomFilesBaseClass, self).__init__(*args, **kwargs)
                # copy attribute
                self.__dict__ = deepcopy(dicom_files.__dict__)
        else:
            super(DicomFilesBaseClass, self).__init__(*args, **kwargs)

    @abstractmethod
    def add_dicom_dataset(self, dicom_file):
        pass

