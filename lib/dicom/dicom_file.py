# I hope I can use this for 2 years
# 2019.12.17 Jiazhou Wang
# Email: wjiazhou@gmail.com

import logging

import pydicom.uid
from pydicom import FileDataset
from pydicom.errors import InvalidDicomError
from pydicom.filereader import dcmread

logger = logging.getLogger(__name__)


def dcm_scan(filename, stop_before_pixels=True):
    # this function was used for scan directory require as fast as possible
    try:
        logger.debug("reading: [{}]".format(filename))
        ds = dcmread(filename, force=True, stop_before_pixels=stop_before_pixels)
    except InvalidDicomError:
        logger.info("[{}] is not a valid dicom file. InvalidDicomError".format(filename))
        return None

    #  The only criteria for decide the DICOM file
    if 'SOPInstanceUID' not in ds:
        logger.debug("[{}] is not a valid dicom file. No SOPInstanceUID".format(filename))
        return None
    else:
        # fix some bug
        reference_frame_fix(ds)
        return ds


def dcm_load(filename):
    # this function was used to load full dicom file
    try:
        logger.debug("reading: [{}]".format(filename))
        ds = dcmread(filename, force=True)
    except InvalidDicomError:
        logger.info("[{}] is not a valid dicom file. InvalidDicomError".format(filename))
        return None

    #  The only criteria for decide the DICOM file
    if 'SOPInstanceUID' not in ds:
        logger.debug("[{}] is not a valid dicom file. No SOPInstanceUID".format(filename))
        return None
    else:
        # fix some bug
        reference_frame_fix(ds)
        TransferSyntaxUID_loss_bug_fix(ds)
        return ds


def reference_frame_fix(ds):
    if 'FrameOfReferenceUID' not in ds:
        if 'ReferencedFrameOfReferenceSequence' in ds:
            if len(ds.ReferencedFrameOfReferenceSequence) == 1:
                ds.FrameOfReferenceUID = ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
        else:
            logger.error("[{}] do not have reference frame.".format(ds.filename))


def TransferSyntaxUID_loss_bug_fix(ds):
    if 'TransferSyntaxUID' not in ds.file_meta:
        logger.debug('using default transfer syntax, patient id is [{}], file is [{}]'.
                     format(ds.PatientID, ds.filename))
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian


class DicomFile(FileDataset):
    """
    rewrite this class
    1. fix some error
    2. provide some shortcut
    """

    def __init__(self, filename):
        ds = dcm_load(filename)
        super(DicomFile, self).__init__(filename, ds, file_meta=ds.file_meta)
