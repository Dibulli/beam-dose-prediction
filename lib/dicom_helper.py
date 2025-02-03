"""
Created by jiazhou on 2017/6/24
Updated by jiazhou on 2018/1/23
Version: 0.0.4
This file is used for read DICOM RT files
"""

import logging
import os
import random

import pydicom
import numpy as np
import cv2
from pydicom.charset import decode as dicom_decode
from scipy.interpolate import interpn
from scipy.ndimage import zoom

logger = logging.getLogger('raw_dataset.dicom')


def rescale_resolution(image):
    new_image = None
    if np.ndim(image) == 3:
        new_image = np.zeros((512, 512, image.shape[2]), dtype=float)
        zoom_par = [512 / image.shape[0], 512 / image.shape[1]]
        for z in range(image.shape[2]):
            new_image[:, :, z] = zoom(image[:, :, z], zoom_par)
    if np.ndim(image) == 4:
        new_image = np.zeros((512, 512, image.shape[2], image.shape[3]), dtype=float)
        zoom_par = [512 / image.shape[0], 512 / image.shape[1]]
        for roi_i in range(image.shape[3]):
            for z in range(image.shape[2]):
                new_image[:, :, z, roi_i] = zoom(image[:, :, z, roi_i], zoom_par)
    return new_image


class DicomParser:
    """Parses DICOM / DICOM RT files."""

    def __init__(self, dataset=None, filename=None):
        if dataset:
            self.ds = dataset
        elif filename:
            try:
                self.ds = pydicom.read_file(filename, force=True)
            except (EOFError, IOError):
                # Raise the error for the calling method to handle
                raise
            else:
                # Sometimes DICOM files may not have headers, but they should always
                # have a SOPClassUID to declare what type of file it is. If the
                # file doesn't have a SOPClassUID, then it probably isn't DICOM.
                if "SOPClassUID" not in self.ds:
                    raise AttributeError
        else:
            raise AttributeError

    """SOP Class and Instance Methods"""

    def get_sop_class_uid(self):
        """Determine the SOP Class UID of the current file."""

        if self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.2':
            return 'rtdose'
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            return 'rtss'
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.5':
            return 'rtplan'
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
            return 'ct'
        else:
            return None

    def get_sop_instance_uid(self):
        """Determine the SOP Class UID of the current file."""
        return self.ds.SOPInstanceUID

    def get_study_info(self):
        """Return the study information of the current file."""
        study = {}
        if 'StudyDescription' in self.ds:
            desc = self.ds.StudyDescription
        else:
            desc = 'No description'
        study['description'] = desc
        # Don't assume that every dataset includes a study UID
        study['uid'] = self.ds.SeriesInstanceUID
        if 'StudyInstanceUID' in self.ds:
            study['uid'] = self.ds.StudyInstanceUID
        if 'SOPClassUID' in self.ds:
            study['SOPClassUID'] = self.ds.SOPClassUID
        return study

    def get_series_info(self):
        """Return the series information of the current file."""
        series = {}
        if 'SeriesDescription' in self.ds:
            desc = self.ds.SeriesDescription
        else:
            desc = 'No description'
        series['description'] = desc
        series['uid'] = self.ds.SeriesInstanceUID
        # Don't assume that every dataset includes a study UID
        series['study'] = self.ds.SeriesInstanceUID
        if 'StudyInstanceUID' in self.ds:
            series['study'] = self.ds.StudyInstanceUID
        if 'FrameofReferenceUID' in self.ds:
            series['referenceframe'] = self.ds.FrameofReferenceUID
        return series

    def get_referenced_series(self):
        """Return the SOP Class UID of the referenced series."""
        if "ReferencedFrameOfReferenceSequence" in self.ds:
            if "RTReferencedStudySequence" in self.ds.ReferencedFrameOfReferenceSequence[0]:
                if "RTReferencedSeriesSequence" in self.ds.ReferencedFrameOfReferenceSequence[
                            0].RTReferencedStudySequence[0]:
                    if "SeriesInstanceUID" in self.ds.ReferencedFrameOfReferenceSequence[
                                0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]:
                        return self.ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[
                            0].RTReferencedSeriesSequence[0].SeriesInstanceUID
        else:
            return ''

    def get_referenced_study(self):
        """Return the SOP Class UID of the referenced study."""
        if "ReferencedStudies" in self.ds:
            if "RefdSOPInstanceUID" in self.ds.ReferencedStudies[0]:
                return self.ds.ReferencedStudies[0].RefdSOPInstanceUID
        else:
            return ''

    def get_frame_of_referenced_uid(self):
        """Determine the Frame of Reference UID of the current file."""
        if 'FrameOfReferenceUID' in self.ds:
            return self.ds.FrameOfReferenceUID
        elif 'ReferencedFrameofReferences' in self.ds:
            return self.ds.ReferencedFrameofReferences[0].FrameofReferenceUID
        else:
            return ''

    def get_referenced_structure_set(self):
        """Return the SOP Class UID of the referenced structure set."""
        if "ReferencedStructureSets" in self.ds:
            return self.ds.ReferencedStructureSets[0].ReferencedSOPInstanceUID
        else:
            return ''

    def get_referenced_rtplan(self):
        """Return the SOP Class UID of the referenced RT plan."""
        if "ReferencedRTPlans" in self.ds:
            return self.ds.ReferencedRTPlans[0].ReferencedSOPInstanceUID
        else:
            return ''

    def get_demographics(self):
        """Return the patient demographics from a DICOM file."""
        # Set up some sensible defaults for demographics
        patient = {'name': 'N/A',
                   'id': 'N/A',
                   'dob': 'None Found',
                   'gender': 'Other'}
        if 'PatientsName' in self.ds:
            if self.decode("PatientsName") == '':
                logger.debug('Patient do not have name!')
            else:
                patient['name'] = self.decode("PatientsName").original_string.replace('^', ', ')
        if 'PatientID' in self.ds:
            patient['id'] = self.ds.PatientID
        if 'PatientsSex' in self.ds:
            if self.ds.PatientsSex == 'M':
                patient['gender'] = 'Male'
            elif self.ds.PatientsSex == 'F':
                patient['gender'] = 'Female'
        if 'PatientsBirthDate' in self.ds:
            if len(self.ds.PatientsBirthDate):
                patient['dob'] = str(self.ds.PatientsBirthDate)
        return patient

    def decode(self, tag):
        """Apply the DICOM character encoding to the given tag."""
        if tag not in self.ds:
            return None
        else:
            oldval = self.ds.data_element(tag).value
            if isinstance(oldval, list):
                oldval = oldval[0]
            try:
                cs = self.ds.get('SpecificCharacterSet', "ISO_IR 6")
                dicom_decode(self.ds.data_element(tag), cs)
            except:
                logger.info("Could not decode character set for %s.", oldval)
                return str(self.ds.data_element(tag).value, errors='replace')
            newval = self.ds.data_element(tag).value
            self.ds.data_element(tag).value = oldval
            return newval

    def get_structures(self):
        """Returns the structures (ROIs) with their coordinates."""

        def __get_contour_points(array):
            """Parses an array of xyz points and returns a array of point dictionaries."""
            return np.array(array).reshape((-1, 3))

        def __cal_plane_thickness(planes_dict):
            """Calculates the plane thickness for each structure."""
            planes_for_cal = []
            # Iterate over each plane in the structure
            for z in planes_dict:
                planes_for_cal.append(float(z))
            planes_for_cal.sort()
            # Determine the thickness
            thickness = 10000
            for n in range(0, len(planes_for_cal)):
                if n > 0:
                    new_thickness = planes_for_cal[n] - planes_for_cal[n - 1]
                    if new_thickness < thickness:
                        thickness = new_thickness
            # If the thickness was not detected, set it to 0
            if thickness == 10000:
                thickness = 0

            return thickness

        structures = {}

        # Determine whether this is RT Structure Set file
        if not (self.get_sop_class_uid() == 'rtss'):
            return structures

        # Locate the name and number of each ROI
        if 'StructureSetROISequence' in self.ds:
            for item in self.ds.StructureSetROISequence:
                data = {}
                number = item.ROINumber
                data['id'] = number
                data['name'] = item.ROIName
                logger.debug("Found ROI %s: %s", str(number), data['name'])
                structures[number] = data

        # Determine the type of each structure (PTV, organ, external, etc)
        if 'RTROIObservationsSequence' in self.ds:
            for item in self.ds.RTROIObservationsSequence:
                number = item.ReferencedROINumber
                structures[number]['RTROIType'] = item.RTROIInterpretedType

        # The coordinate data of each ROI is stored within ROIContourSequence
        if 'ROIContourSequence' in self.ds:
            for roi in self.ds.ROIContourSequence:
                number = roi.ReferencedROINumber

                # Generate a random color for the current ROI
                structures[number]['color'] = np.array((
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)), dtype=float)
                # Get the RGB color triplet for the current ROI if it exists
                if 'ROIDisplayColor' in roi:
                    # Make sure the color is not none
                    if roi.ROIDisplayColor is not None:
                        color = roi.ROIDisplayColor
                    # Otherwise decode values separated by forward slashes
                    else:
                        value = roi[0x3006, 0x002a].repval
                        color = value.strip("'").split("/")
                    # Try to convert the detected value to a color triplet
                    try:
                        structures[number]['color'] = np.array(color, dtype=float)
                    # Otherwise fail and fallback on the random color
                    except:
                        logger.debug("Unable to decode display color for ROI #%s", str(number))

                planes = {}
                if 'ContourSequence' in roi:
                    # Locate the contour sequence for each referenced ROI
                    for contour in roi.ContourSequence:
                        # For each plane, initialize a new plane dictionary
                        plane = {'geometricType': contour.ContourGeometricType,
                                 'numContourPoints': contour.NumberOfContourPoints,
                                 'contourData': __get_contour_points(contour.ContourData)}

                        # Determine all the plane properties

                        # Each plane which coincides with a image slice will have a unique ID
                        if 'ContourImages' in contour:
                            plane['RefUID'] = contour.ContourImages[0].ReferencedSOPInstanceUID

                        # Each plane which coincides with a image slice will have a unique ID
                        if 'ContourImageSequence' in contour:
                            plane['RefUID'] = contour.ContourImageSequence[0].ReferencedSOPInstanceUID

                        if 'RefUID' not in plane:
                            plane['RefUID'] = []

                        # Add each plane to the planes dictionary of the current ROI
                        if 'geometricType' in plane:
                            z = str('%.2f' % list(plane['contourData'])[0][2]).replace('-0', '0')
                            if z not in planes:
                                planes[z] = []
                            planes[z].append(plane)

                # Calculate the plane thickness for the current ROI
                structures[number]['thickness'] = __cal_plane_thickness(planes)

                # Add the planes dictionary to the current ROI
                structures[number]['planes'] = planes

        return structures

    def get_dose_info(self):
        """Return the dose data from a DICOM RT Dose file."""
        dose_info = {'doseunits': self.ds.DoseUnits,
                     'dosetype': self.ds.DoseType,
                     'dosesummationtype': self.ds.DoseSummationType,
                     'dosegridscaling': self.ds.DoseGridScaling,
                     'dosemax': float(self.ds.pixel_array.max())}
        return dose_info

    def get_plan(self):
        """Returns the plan information."""

        self.plan = {}

        self.plan['label'] = self.ds.RTPlanLabel
        self.plan['date'] = self.ds.RTPlanDate
        self.plan['time'] = self.ds.RTPlanTime
        self.plan['name'] = ''
        self.plan['rxdose'] = 0
        if "DoseReferences" in self.ds:
            for item in self.ds.DoseReferences:
                if item.DoseReferenceStructureType == 'SITE':
                    self.plan['name'] = "N/A"
                    if "DoseReferenceDescription" in item:
                        self.plan['name'] = item.DoseReferenceDescription
                    if item.has_key('TargetPrescriptionDose'):
                        rxdose = item.TargetPrescriptionDose * 100
                        if (rxdose > self.plan['rxdose']):
                            self.plan['rxdose'] = rxdose
                elif item.DoseReferenceStructureType == 'VOLUME':
                    if 'TargetPrescriptionDose' in item:
                        self.plan['rxdose'] = item.TargetPrescriptionDose * 100
        if (("FractionGroups" in self.ds) and (self.plan['rxdose'] == 0)):
            fg = self.ds.FractionGroups[0]
            if ("ReferencedBeams" in fg) and ("NumberofFractionsPlanned" in fg):
                beams = fg.ReferencedBeams
                fx = fg.NumberofFractionsPlanned
                for beam in beams:
                    if "BeamDose" in beam:
                        self.plan['rxdose'] += beam.BeamDose * fx * 100
        self.plan['rxdose'] = int(self.plan['rxdose'])
        return self.plan

    def get_referenced_beams_in_fraction(self, fx=0):
        """Return the referenced beams from the specified fraction."""

        beams = {}
        if ("Beams" in self.ds):
            bdict = self.ds.Beams
        elif ("IonBeams" in self.ds):
            bdict = self.ds.IonBeams
        else:
            return beams

        # Obtain the beam information
        for b in bdict:
            beam = {}
            beam['name'] = b.BeamName if "BeamName" in b else ""
            beam['description'] = b.BeamDescription \
                if "BeamDescription" in b else ""
            beams[b.BeamNumber] = beam

        # Obtain the referenced beam info from the fraction info
        if ("FractionGroups" in self.ds):
            fg = self.ds.FractionGroups[fx]
            if ("ReferencedBeams" in fg):
                rb = fg.ReferencedBeams
                nfx = fg.NumberofFractionsPlanned
                for b in rb:
                    if "BeamDose" in b:
                        beams[b.ReferencedBeamNumber]['dose'] = \
                            b.BeamDose * nfx * 100
        return beams


class DicomDirectory:
    """
    Scan DICOM directory
    """

    def __init__(self, path):
        self.__path = path

        # this four index was used for fast localization
        self.__uid2patient = {}
        self.__uid2series = {}
        self.__uid2studies = {}
        self.__uid2file = {}

        # patient id index
        self.__patient_id_to_index = {}
        self.__index_to_patient_id = []

        # tmp variable
        self.__curr_patient = {}
        self.__curr_patient_index = 0
        self.__curr_study_list = []
        self.__curr_study = {}
        self.__curr_study_index = 0
        self.__curr_series_list = []
        self.__curr_series = {}
        self.__curr_series_index = 0
        self.__curr_file_list = []
        self.__curr_file_index = 0

        # index for loop
        self.__all_image_series = []
        self.__uid_2_index_of_all_image_series = {}
        self.__all_rtst_file = []
        self.__all_rtst_uid = []
        self.__uid_2_index_of_all_rtst = {}
        self.__all_dose_file = []
        self.__all_dose_uid = []
        self.__uid_2_index_of_all_dose = {}
        self.__all_plan_file = []
        self.__all_plan_uid = []
        self.__uid_2_index_of_all_plan = {}

        # final data set
        self.data_set = []
        self.scan()

    # check path
    def __check_path_valid(self):
        if os.path.isdir(self.__path):
            pass
        else:
            logger.debug('It is not a valid location.')

    # sub method of scan
    def __is_patient_in_data_set(self, patient_id):
        if patient_id in self.__patient_id_to_index:
            return True
        else:
            return False

    # sub method of __add_series_info
    def __is_series_in_data_set(self, uid):
        if uid in self.__uid2series:
            return True
        else:
            return False

    # sub method of __add_study_info
    def __is_study_in_data_set(self, uid):
        if uid in self.__uid2studies:
            return True
        else:
            return False

    # sub method of __add_file_info
    def __is_file_in_data_set(self, uid):
        if uid in self.__uid2file:
            return True
        else:
            return False

    # sub method of __add_series_info
    def __find_series_index_by_uid(self, uid):
        return self.__uid2series[uid]

    # sub method of __add_study_info
    def __find_study_index_by_uid(self, uid):
        return self.__uid2studies[uid]

    # sub method of __add_file_info
    def __find_file_index_by_uid(self, uid):
        return self.__uid2file[uid]

    # sub method of scan
    def __get_all_files_in_this_folder(self):
        files_list = []
        if os.path.isdir(self.__path):
            # get all files' path from directory
            for root, subdirs, filenames in os.walk(self.__path):
                files_list += map(lambda f: os.path.join(root, f), filenames)
        return files_list

    # sub method of __add_info_to_data_set
    def __add_study_info(self, study_uid, file_name):
        if self.__is_study_in_data_set(study_uid):
            # find study and set study pointer
            logger.debug('Find the study for file.' + file_name)
            # find the index of the study in this patient
            curr_study_index = self.__find_study_index_by_uid(study_uid)
            self.__curr_study_index = curr_study_index
            # TODO：Here need add a consistent check
            # set current study
            try:
                self.__curr_study = self.__curr_patient['study_list'][curr_study_index]
            except KeyError:
                print(self.__curr_patient)
        else:
            # add new study into whole data structure
            logger.debug('Not find the study for file.' + file_name)
            logger.debug('Add new study.')
            # create a new study
            curr_study = {'study_uid': study_uid}
            if 'study_list' not in self.__curr_patient:
                self.__curr_patient['study_list'] = []
            # add to current patient
            self.__curr_patient['study_list'].append(curr_study)
            # find the index of study in this patient
            num_of_studies_in_current_patient = len(self.__curr_patient['study_list'])
            curr_study_index = num_of_studies_in_current_patient - 1
            self.__curr_study_index = curr_study_index
            # set current study
            self.__curr_study = self.__curr_patient['study_list'][curr_study_index]
            # write the quick index
            self.__uid2studies[study_uid] = curr_study_index
            self.__uid2patient[study_uid] = self.__curr_patient_index

    # sub method of __add_info_to_data_set
    def __add_series_info(self, series_uid, modality, frame_of_ref, file_name):
        if self.__is_series_in_data_set(series_uid):
            # find series and set series pointer
            logger.debug('Find the series for file.' + file_name)
            # find the index of the series in this study
            curr_series_index = self.__find_series_index_by_uid(series_uid)
            self.__curr_series_index = curr_series_index
            # TODO：Here need add a consistent check
            # set current series
            self.__curr_series = self.__curr_study['series_list'][curr_series_index]
        else:
            # add new study into whole data structure
            logger.debug('Not find the series for file.' + file_name)
            logger.debug('Add new series.')
            # create a new series
            curr_series = {'series_uid': series_uid,
                           'modality': modality,
                           'frame_of_ref': frame_of_ref}
            if 'series_list' not in self.__curr_study:
                self.__curr_study['series_list'] = []
            # add to current patient
            self.__curr_study['series_list'].append(curr_series)

            # find the index of series in this study
            num_of_series_in_current_study = len(self.__curr_study['series_list'])
            curr_series_index = num_of_series_in_current_study - 1
            self.__curr_series_index = curr_series_index
            # set current series
            self.__curr_series = self.__curr_study['series_list'][curr_series_index]
            # write the quick index
            self.__uid2series[series_uid] = curr_series_index
            self.__uid2studies[series_uid] = self.__curr_study_index
            self.__uid2patient[series_uid] = self.__curr_patient_index

    # sub method of __add_info_to_data_set
    def __add_file_info(self, file_uid, file_name):
        # process file info
        if self.__is_file_in_data_set(file_uid):
            # should not have 2 same UID DICOM file
            logger.warning('Find same UID for file.' + file_name)
        else:
            # add new study into whole data structure
            logger.debug('This is a new file.' + file_name)
            # create a new file
            curr_file = {'file_uid': file_uid,
                         'file_name': file_name}
            if 'file_list' not in self.__curr_series:
                self.__curr_series['file_list'] = []
            # add to current file list
            self.__curr_series['file_list'].append(curr_file)

            # find the index of file in this series
            num_of_files_in_current_series = len(self.__curr_series['file_list'])
            curr_file_index = num_of_files_in_current_series - 1
            # set current file list
            self.__curr_file = self.__curr_series['file_list'][curr_file_index]
            # write the quick index
            self.__uid2file[file_uid] = curr_file_index
            self.__uid2series[file_uid] = self.__curr_series_index
            self.__uid2studies[file_uid] = self.__curr_study_index
            self.__uid2patient[file_uid] = self.__curr_patient_index

    # a general method used to establish data set tree structure
    def __add_info_to_data_set(self, dp, file_name):
        # get info from file
        series_info = dp.get_series_info()
        study_info = dp.get_study_info()
        series_uid = series_info['uid']
        study_uid = study_info['uid']
        file_uid = dp.get_sop_instance_uid()
        frame_of_ref = dp.get_frame_of_referenced_uid()
        modality = dp.ds.SOPClassUID.name

        # process study info
        self.__add_study_info(study_uid, file_name)
        # process series info
        self.__add_series_info(series_uid, modality, frame_of_ref, file_name)
        # process file info
        self.__add_file_info(file_uid, file_name)

    # a method to establish data set tree structure for image
    def __add_image_info_to_data_set(self, dp, file_name):
        # add general info
        self.__add_info_to_data_set(dp, file_name)
        # other information, for image only add its series uid to index
        series_info = dp.get_series_info()
        series_uid = series_info['uid']
        if series_uid in self.__uid_2_index_of_all_image_series:
            # have recoded it, just pass
            pass
        else:
            self.__all_image_series.append(series_uid)
            self.__uid_2_index_of_all_image_series[series_uid] = len(self.__all_image_series) - 1

    # a method to establish data set tree structure for rtst
    def __add_rtst_info_to_data_set(self, dp, file_name):
        file_uid = dp.get_sop_instance_uid()
        # check file whether exist in data set
        if file_uid in self.__uid_2_index_of_all_rtst:
            # have recoded it, give warining
            logger.warning('Find same UID for file.' + file_name)
        else:
            # add general info
            self.__add_info_to_data_set(dp, file_name)
            # add rtst information
            self.__curr_file['ref_series'] = dp.get_referenced_series()
            # add its file uid to loop index
            self.__all_rtst_file.append(file_name)
            self.__all_rtst_uid.append(file_uid)
            self.__uid_2_index_of_all_rtst[file_uid] = len(self.__all_rtst_uid) - 1

    # a method to establish data set tree structure for plan
    def __add_rtplan_info_to_data_set(self, dp, file_name):
        file_uid = dp.get_sop_instance_uid()
        # check file whether exist in data set
        if file_uid in self.__uid_2_index_of_all_plan:
            # have recoded it, give warining
            logger.warning('Find same UID for file.' + file_name)
        else:
            # first add general info
            self.__add_info_to_data_set(dp, file_name)
            # add its file uid to loop index
            self.__all_plan_file.append(file_name)
            self.__all_plan_uid.append(file_uid)
            self.__uid_2_index_of_all_plan[file_uid] = len(self.__all_plan_uid) - 1

    # a method to establish data set tree structure for dose
    def __add_rtdose_info_to_data_set(self, dp, file_name):
        file_uid = dp.get_sop_instance_uid()
        # check file whether exist in data set
        if file_uid in self.__uid_2_index_of_all_dose:
            # have recoded it, give warining
            logger.warning('Find same UID for file.' + file_name)
        else:
            # first add general info
            self.__add_info_to_data_set(dp, file_name)
            # add rtst information
            self.__curr_file['ref_study'] = dp.get_referenced_study()
            # other information, for dose only add its file uid to index
            self.__all_dose_file.append(file_name)
            self.__all_dose_uid.append(file_uid)
            self.__uid_2_index_of_all_dose[file_uid] = len(self.__all_dose_uid) - 1

    def scan(self):
        self.__check_path_valid()
        files_list = self.__get_all_files_in_this_folder()

        # Loop all files in this directory
        for file_name in files_list:
            if os.path.isfile(file_name):
                try:
                    logger.debug("Reading: %s", file_name)
                    logger.debug("Scanning: " + file_name)
                    dp = DicomParser(filename=file_name)
                except (AttributeError, EOFError, IOError, KeyError):
                    logger.debug("%s is not a valid DICOM file.", file_name)
                else:

                    # base demographics part
                    base_demographics = dp.get_demographics()
                    # get patient's name, id, dob and gender
                    patient_id = base_demographics['id']
                    logger.debug('This file belong to: ' + patient_id)
                    if self.__is_patient_in_data_set(patient_id):
                        # find patient and set patient pointer
                        curr_patient_index = self.__patient_id_to_index[patient_id]
                        self.__curr_patient = self.data_set[curr_patient_index]
                        self.__curr_patient_index = curr_patient_index
                    else:
                        # add new patient into whole data structure
                        self.__curr_patient = {'demographics': base_demographics}
                        self.data_set.append(self.__curr_patient)
                        self.__index_to_patient_id.append(patient_id)
                        self.__curr_patient_index = len(self.__index_to_patient_id) - 1
                        self.__patient_id_to_index[patient_id] = self.__curr_patient_index

                    # image file part
                    # to include all dicom image, CT, MR and PET
                    if ('ImageOrientationPatient' in dp.ds) and not dp.get_sop_class_uid() == 'rtdose':
                        self.__add_image_info_to_data_set(dp, file_name)
                        self.data_set[self.__curr_patient_index] = self.__curr_patient
                    # structure file part
                    elif dp.ds.Modality == 'RTSTRUCT':
                        self.__add_rtst_info_to_data_set(dp, file_name)
                        self.data_set[self.__curr_patient_index] = self.__curr_patient
                    # plan file part
                    elif dp.ds.Modality == 'RTPLAN':
                        self.__add_rtplan_info_to_data_set(dp, file_name)
                        self.data_set[self.__curr_patient_index] = self.__curr_patient
                    # dose file part
                    elif dp.ds.Modality == 'RTDOSE':
                        self.__add_rtdose_info_to_data_set(dp, file_name)
                        self.data_set[self.__curr_patient_index] = self.__curr_patient
                    # Otherwise it is a currently unsupported file
                    else:
                        logger.info("%s is a %s file and is not currently supported.",
                                    file_name, dp.ds.SOPClassUID.name)
        # provide a feedback
        progress_str = 'Find ' + str(len(self.data_set)) + ' patients. \n'
        progress_str = progress_str + 'Find ' + str(len(self.get_all_image_series())) + ' image series. \n'
        progress_str = progress_str + 'Find ' + str(len(self.get_all_rtst_uid())) + ' rtst files. \n'
        progress_str = progress_str + 'Find ' + str(len(self.get_all_dose_file())) + ' dose files. \n'
        logger.info(progress_str)

    def get_path(self):
        return self.__path

    def find_file_by_uid(self, uid):
        return self.__uid2file[uid]

    def get_all_image_series(self):
        return self.__all_image_series

    def get_all_rtst_uid(self):
        return self.__all_rtst_file

    def get_all_plan_uid(self):
        return self.__all_plan_file

    def get_all_dose_uid(self):
        return self.__all_dose_uid

    def get_all_dose_file(self):
        return self.__all_dose_file

    def get_all_img_files_name_by_img_series_uid(self, uid):
        if uid in self.__uid_2_index_of_all_image_series:
            patient_i = self.__uid2patient[uid]
            study_i = self.__uid2studies[uid]
            series_i = self.__uid2series[uid]
            file_list = []
            try:
                file_list = self.data_set[patient_i]['study_list'][study_i]['series_list'][series_i]['file_list']
            except IndexError:
                self.print_tree('file_tree.txt')
                logger.error('patient_i = ' + str(patient_i))
                logger.error('study_i = ' + str(study_i))
                logger.error('series_i = ' + str(series_i))
            file_name_list = []
            for file in file_list:
                file_name_list.append(file['file_name'])
            return file_name_list
        else:
            logger.warning('Not find this image series' + uid)
            return 0

    def get_dose_file_by_rtst_uid(self, uid):
        if uid in self.__uid_2_index_of_all_rtst:
            patient_i = self.__uid2patient[uid]
            study_i = self.__uid2studies[uid]
            series_i = self.__uid2series[uid]
            file_i = self.__uid2file[uid]
            rtdose_series_num = 0
            rtdose_series_index = 0
            for series_i, series in enumerate(self.data_set[patient_i]
                                              ['study_list'][self.__uid2studies[uid]]['series_list']):
                if series['modality'] == 'RT Dose Storage':
                    rtdose_series_num = rtdose_series_num + 1
                    rtdose_series_index = series_i
            if rtdose_series_num == 0:
                logger.warning('Not find dose file.')
                return 0
            if rtdose_series_num > 1:
                logger.warning('More than one dose file.')
                return 0
            rtdose_file = self.data_set[patient_i]['study_list'][study_i]['series_list'][
                rtdose_series_index]['file_list'][0]['file_name']
            return rtdose_file

    def get_ref_img_files_by_dose_uid(self, uid):
        if uid not in self.__uid_2_index_of_all_dose:
            logger.warning('Not find this dose uid ' + uid)
            return 0
        patient_i = self.__uid2patient[uid]
        study_i = self.__uid2studies[uid]
        series_i = self.__uid2series[uid]
        file_i = self.__uid2file[uid]
        # find the study uid of the dose
        img_study_uid = \
            self.data_set[patient_i]['study_list'][study_i]['series_list'][
                series_i]['file_list'][file_i]['ref_study']
        dose_frame_of_ref = self.data_set[patient_i]['study_list'][
            study_i]['series_list'][series_i]['frame_of_ref']

        # find referenced image study
        image_patient_i = self.__uid2patient[img_study_uid]  # in case patient ID is different
        image_study_i = self.__uid2studies[img_study_uid]

        # two criteria: 1. same frame of ref
        #               2. only have one image series in this study
        image_series_num = 0
        image_series_index = 0
        for series_i, series in enumerate(self.data_set[image_patient_i][
                                              'study_list'][image_study_i][
                                              'series_list']):
            if series['modality'] == 'CT Image Storage' \
                    or series['modality'] == 'MR Image Storage' \
                    or series['modality'] == 'PET Image Storage':
                image_series_num = image_series_num + 1
                image_series_index = series_i
        if image_series_num > 1:
            logger.warning('This dose have more than one image series. Can not decide !!!')
            return 0
        img_series = self.data_set[patient_i]['study_list'][study_i][
            'series_list'][image_series_index]
        if dose_frame_of_ref != img_series['frame_of_ref']:
            logger.warning('frame of reference ot match !!!')
            return 0
        return self.get_all_img_files_name_by_img_series_uid(img_series['series_uid'])

    def get_ref_img_files_by_rtst_uid(self, uid):
        if uid not in self.__uid_2_index_of_all_rtst:
            logger.warning('Not find this rtst uid ' + uid)
            return 0
        patient_i = self.__uid2patient[uid]
        study_i = self.__uid2studies[uid]
        series_i = self.__uid2series[uid]
        file_i = self.__uid2file[uid]
        # find the series uid of the rtst
        img_series_uid = \
            self.data_set[patient_i]['study_list'][study_i]['series_list'][
                series_i]['file_list'][file_i]['ref_series']

        return self.get_all_img_files_name_by_img_series_uid(img_series_uid)

    def get_rtst_file_by_dose_uid(self, uid):
        if uid not in self.__uid_2_index_of_all_dose:
            logger.warning('Not find this dose uid' + uid)
            return 0

        patient_i = self.__uid2patient[uid]
        study_i = self.__uid2studies[uid]
        series_i = self.__uid2series[uid]
        file_i = self.__uid2file[uid]
        # find the study uid of the dose
        img_study_uid = self.data_set[patient_i]['study_list'][study_i]['series_list'][series_i]['file_list'][file_i]['ref_study']
        dose_frame_of_ref = self.data_set[patient_i]['study_list'][
            study_i]['series_list'][
            series_i]['frame_of_ref']

        # find rtst file in this study
        rtst_patient_i = self.__uid2patient[img_study_uid]  # in case patient ID is different
        rtst_study_i = self.__uid2studies[img_study_uid]

        # two criteria: 1. same frame of ref
        #               2. only have one rtst file in this study

        # count
        rtst_series_num = 0
        rtst_series_index = 0
        for series_i, series in enumerate(self.data_set[rtst_patient_i][
                                              'study_list'][rtst_study_i][
                                              'series_list']):
            if series['modality'] == 'RT Structure Set Storage':
                rtst_series_num = rtst_series_num + 1
                rtst_series_index = series_i
        # Not find
        if rtst_series_num == 0:
            logger.warning('Not find the rtst series.')
            return 0
        # more than one series
        if rtst_series_num > 1:
            logger.warning('This dose have more than one rtst series. Can not decide !!!')
            return 0
        rtst_series = self.data_set[patient_i]['study_list'][study_i][
            'series_list'][rtst_series_index]
        # count
        rtst_file_num = len(rtst_series['file_list'])
        # Not find
        if rtst_file_num == 0:
            logger.warning('Not find the rtst file.')
            return 0
        # more than one files
        if rtst_file_num > 1:
            logger.warning('This dose have more than one rtst file. Can not decide !!!')
            return 0
        # here return the file name
        rtst_file = self.data_set[patient_i]['study_list'][study_i][
            'series_list'][rtst_series_index]['file_list'][0]['file_name']
        if dose_frame_of_ref != rtst_series['frame_of_ref']:
            logger.warning('frame of reference ot match !!!')
            return 0
        return rtst_file

    def get_plan_file_by_dose_uid(self, uid):
        if uid not in self.__uid_2_index_of_all_dose:
            logger.warning('Not find this dose uid' + uid)
            return 0

        patient_i = self.__uid2patient[uid]
        study_i = self.__uid2studies[uid]
        series_i = self.__uid2series[uid]
        file_i = self.__uid2file[uid]
        # find the plan uid of the dose
        plan_uid = self.data_set[patient_i]['study_list'][study_i][
            'series_list'][series_i]['file_list'][file_i]['ref_plan']

        if plan_uid not in self.__uid_2_index_of_all_plan:
            logger.warning('Not find this plan uid ' + uid)
            return 0

        # check or not?
        patient_i = self.__uid2patient[plan_uid]
        study_i = self.__uid2studies[plan_uid]
        series_i = self.__uid2series[plan_uid]
        file_i = self.__uid2file[plan_uid]
        # find the series uid of the rtst
        plan_file = self.data_set[patient_i]['study_list'][study_i]['series_list'][
                series_i]['file_list'][file_i]['file_name']

        return plan_file

    def print_tree(self, filename='tree.txt'):
        def print_dict(dict_ins, indent):
            str_dict = ''
            for key in dict_ins:
                if isinstance(dict_ins[key], str):
                    str_dict = str_dict + '\t' * indent + key + '--' + dict_ins[key] + '\n'
            return str_dict

        output_file = open(filename, "w")
        new_line = ''
        for pt_index, pt in enumerate(self.data_set):
            new_line = new_line + 'patient\t' + str(pt_index) + '\n'
            new_line = new_line + print_dict(pt['demographics'], 1)
            for study_index, study in enumerate(pt['study_list']):
                new_line = new_line + '\tstudy\t' + str(study_index) + '\n'
                new_line = new_line + print_dict(study, 2)
                new_line = new_line + '\t\tuid to patient \t' + str(self.__uid2patient[study['study_uid']]) + '\n'
                new_line = new_line + '\t\tuid to study \t' + str(self.__uid2studies[study['study_uid']]) + '\n'
                for series_index, series in enumerate(study['series_list']):
                    new_line = new_line + '\t\tseries\t' + str(series_index) + '\n'
                    new_line = new_line + print_dict(series, 3)
                    new_line = new_line + '\t\t\tuid to patient \t' + str(
                        self.__uid2patient[series['series_uid']]) + '\n'
                    new_line = new_line + '\t\t\tuid to study \t' + str(self.__uid2studies[series['series_uid']]) + '\n'
                    new_line = new_line + '\t\t\tuid to series \t' + str(self.__uid2series[series['series_uid']]) + '\n'
                    for file_index, file in enumerate(series['file_list']):
                        new_line = new_line + '\t\t\tfile\t' + str(file_index) + '\n'
                        new_line = new_line + print_dict(file, 4)
        output_file.write(new_line)
        output_file.close()


class AffineMatrix:
    def __init__(self, ds):
        self.delta_r = 0
        self.delta_c = 0
        self.vox2abs = np.zeros((3, 3), dtype=np.float)
        self.abs2vox = np.zeros((3, 3), dtype=np.float)
        self.vox2abs_4D = np.zeros((4, 4), dtype=np.float)
        self.abs2vox_4D = np.zeros((4, 4), dtype=np.float)

        self.__gen_affine_vox2abs(ds)
        self.__gen_affine_abs2vox()
        self.__gen_affine_vox2abs_4d(ds)
        self.__gen_affine_abs2vox_4d()

    def __gen_affine_vox2abs(self, ds):
        delta_r = ds.PixelSpacing[0]
        delta_c = ds.PixelSpacing[1]
        T11 = ds.ImagePositionPatient[0]
        T12 = ds.ImagePositionPatient[1]
        T13 = ds.ImagePositionPatient[2]
        image_orientation = ds.ImageOrientationPatient
        F11 = image_orientation[0]
        F21 = image_orientation[1]
        F31 = image_orientation[2]
        F12 = image_orientation[3]
        F22 = image_orientation[4]
        F32 = image_orientation[5]

        self.vox2abs = np.matrix(
            [[F11 * delta_r, F12 * delta_c, T11],
             [F21 * delta_r, F22 * delta_c, T12],
             [F31 * delta_r, F32 * delta_c, T13]])
        if np.linalg.matrix_rank(self.vox2abs) < 3:
            logger.warning('patient ID: ' + ds.PatientID)
            logger.warning('This affine matrix rank is less than 3!')
            self.vox2abs[2, 2] = 1

    def __gen_affine_vox2abs_4d(self, ds):
        delta_r = ds.PixelSpacing[0]
        delta_c = ds.PixelSpacing[1]
        delta_z = delta_r
        T11 = ds.ImagePositionPatient[0]
        T12 = ds.ImagePositionPatient[1]
        T13 = ds.ImagePositionPatient[2]
        image_orientation = ds.ImageOrientationPatient
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

        self.vox2abs_4D = np.matrix(
            [[F11 * delta_r, F12 * delta_c, F13 * delta_z, T11],
             [F21 * delta_r, F22 * delta_c, F23 * delta_z, T12],
             [F31 * delta_r, F32 * delta_c, F33 * delta_z, T13],
             [0, 0, 0, 1]])

    def __gen_affine_abs2vox(self):
        self.abs2vox = np.linalg.inv(self.vox2abs)

    def __gen_affine_abs2vox_4d(self):
        self.abs2vox_4D = np.linalg.inv(self.vox2abs_4D)


class DicomImage:
    def __init__(self, filearray=None):

        self.patient_id = None
        self.patient_name = None
        self.patient_sex = None
        self.patient_age = None
        self.study_data = None
        self.study_time = None

        self.rows = 0
        self.columns = 0
        self.z_length = 0
        self.pixel_space = [0, 0]
        self.slice_thickness = 0
        self.PatientPosition = 'hfs'

        self.image_2d_raw = []
        self.image_3d_raw = None
        self.image_3d_norm = []

        self.rescale_intercept = 0
        self.rescale_slope = 1
        self.abs_z_list = []
        self.__uid2z = {}
        self.__affine_mat_dict = {}

        self.frame_of_ref_uid = None
        self.study_uid = None
        self.series_uid = None
        self.image_uid_list = []
        self.pixel_volume = 0

        self.loc_x = []
        self.loc_y = []
        self.loc_z = []

        self.load_image(filearray=filearray)
        self.create_3D_image()

    def create_3D_image(self):
        # initial 3d grid
        # create 3d grid
        self.image_3d_raw = np.zeros((self.rows, self.columns, self.z_length))
        for z, current_image in enumerate(self.image_2d_raw):
            try:
                self.image_3d_raw[:, :, z] = current_image.pixel_array * self.rescale_slope + self.rescale_intercept
            except ValueError:
                logger.error(self.patient_id + ' have error in reading pixel array from dicom file!')
            except AttributeError:
                logger.warning(self.patient_id + ' do not have TransferSyntaxUID!')
                current_image.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                self.image_3d_raw[:, :, z] = current_image.pixel_array * self.rescale_slope + self.rescale_intercept

            self.__uid2z[current_image.SOPInstanceUID] = z
            self.image_uid_list.append(current_image.SOPInstanceUID)
        logger.debug('3D image created!')

    def read_base_info(self, dp):
        # Load base information
        self.patient_id = dp.ds.PatientID
        self.patient_name = dp.ds.PatientName
        self.patient_sex = dp.ds.PatientSex
        if hasattr(dp.ds, 'PatientAge'):
            self.patient_age = dp.ds.PatientAge
        else:
            self.patient_age = '50'
        self.study_data = dp.ds.StudyDate
        self.study_time = dp.ds.StudyTime
        self.rows = dp.ds.Rows
        self.columns = dp.ds.Columns
        self.pixel_space = (float(dp.ds.PixelSpacing[0]), float(dp.ds.PixelSpacing[0]))
        self.slice_thickness = float(dp.ds.SliceThickness)
        self.pixel_volume = dp.ds.PixelSpacing[0] * dp.ds.PixelSpacing[1] * dp.ds.SliceThickness
        if hasattr(dp.ds, 'RescaleIntercept'):
            self.rescale_intercept = dp.ds.RescaleIntercept
        else:
            self.rescale_intercept = 0
        if hasattr(dp.ds, 'RescaleSlope'):
            self.rescale_slope = dp.ds.RescaleSlope
        else:
            self.rescale_slope = 1
        self.frame_of_ref_uid = dp.get_frame_of_referenced_uid()
        self.study_uid = dp.get_study_info()['uid']
        self.series_uid = dp.get_series_info()['uid']
        if hasattr(dp.ds, 'PatientPosition'):
            self.PatientPosition = dp.ds.PatientPosition

    def sort_image(self):
        def __is_image_orientation_same(iop0, iop1):
            delta_iop = iop0 - iop1
            if np.any(np.array(np.round(delta_iop.astype(np.double)), dtype=np.int32)):
                return False
            else:
                return True

        def __ipp_sort():
            for image in images:
                unsorted_IPP_z_list.append(image.ImagePositionPatient[2])
            # Sort image numbers in descending order for head first patients
            if hasattr(image, 'PatientPosition'):
                if 'hf' in image.PatientPosition.lower():
                    sorted_index = sorted(range(len(unsorted_IPP_z_list)),
                                          key=lambda k: unsorted_IPP_z_list[k],
                                          reverse=True)
                # Otherwise sort image numbers in ascending order
                else:
                    sorted_index = sorted(range(len(unsorted_IPP_z_list)),
                                          key=lambda k: unsorted_IPP_z_list[k])
            else:
                sorted_index = sorted(range(len(unsorted_IPP_z_list)),
                                      key=lambda k: unsorted_IPP_z_list[k])
            sorted_images = [images[i] for i in sorted_index]
            abs_z_list = [unsorted_IPP_z_list[i] for i in sorted_index]
            return sorted_images, abs_z_list

        # Sort the images based on a sort descriptor:
        # (ImagePositionPatient, InstanceNumber or AcquisitionNumber)
        sortedimages = []
        unsorted_IPP_z_list = []
        images = self.image_2d_raw
        sort = 'IPP'
        # Determine if all images in the series are parallel
        # by testing for differences in ImageOrientationPatient
        parallel = True
        for i, item in enumerate(images):
            if i > 0:
                iop0 = np.array(item.ImageOrientationPatient)
                iop1 = np.array(images[i - 1].ImageOrientationPatient)
                if not __is_image_orientation_same(iop0, iop1):
                    parallel = False
                    break
        # If the images are parallel, sort by ImagePositionPatient
        if parallel:
            sort = 'IPP'
        else:
            # Otherwise sort by Instance Number
            if images[0].InstanceNumber != images[1].InstanceNumber:
                sort = 'InstanceNumber'
                # ! no code for this method !!!
            # Otherwise sort by Acquisition Number
            elif images[0].AcquisitionNumber != images[1].AcquisitionNumber:
                sort = 'AcquisitionNumber'
                # ! no code for this method !!!
        if sort == 'IPP':
            self.image_2d_raw, self.abs_z_list = __ipp_sort()

    def load_image(self, filearray):
        # Get the data from directory.
        logger.debug('Importing patient CT image. Please wait...')

        for n in range(0, len(filearray)):
            dcmfile = str(filearray[n])
            dp = DicomParser(filename=dcmfile)
            if n == 0:
                logger.debug('patient ID: ' + dp.ds.PatientID)
                self.read_base_info(dp)
            # Read the raw data
            self.image_2d_raw.append(dp.ds)

        # Save image z-axis length
        self.z_length = len(self.image_2d_raw)

        # sort image
        self.sort_image()

    def get_z_by_image_file_uid(self, img_file_uid):
        try:
            return self.__uid2z[img_file_uid]
        except KeyError:
            logger.error('Error in patient: ' + self.patient_id)
            logger.error('Not find the slice for this UID', img_file_uid)

    def get_affine_mat(self, z):
        def _if_this_mat_exist():
            if z in self.__affine_mat_dict:
                return True
            else:
                return False

        if _if_this_mat_exist():
            return self.__affine_mat_dict[z]
        else:
            self.__affine_mat_dict[z] = AffineMatrix(self.image_2d_raw[z])
            return self.__affine_mat_dict[z]

    # not precise algorithm
    def get_img_mesh(self, z):
        affine_mat = self.get_affine_mat(z)
        vox2abs = affine_mat.vox2abs
        x = []
        y = []
        for i in range(0, self.rows):
            imat = vox2abs * np.matrix([[i], [0], [1]])
            x.append(float(imat[0]))
        for j in range(0, self.columns):
            jmat = vox2abs * np.matrix([[0], [j], [1]])
            y.append(float(jmat[1]))
        x, y = np.meshgrid(np.array(x), np.array(y))
        return x, y

    # contour abs2rel location
    def points_abs2rel(self, imageUID, contour_abs):
        z = self.get_z_by_image_file_uid(imageUID)
        affine_mat = self.get_affine_mat(z)
        abs2vox = affine_mat.abs2vox
        contour_rel = abs2vox * np.transpose(contour_abs)
        return contour_rel

    # very precise algorithm
    def gen_loc_matrix(self):
        self.loc_x = np.zeros((self.rows, self.columns, self.z_length))
        self.loc_y = np.zeros((self.rows, self.columns, self.z_length))
        self.loc_z = np.zeros((self.rows, self.columns, self.z_length))
        for z, abs_z in enumerate(self.abs_z_list):
            affine_mat = self.get_affine_mat(z)
            vox2abs = affine_mat.vox2abs
            xi_mat, yi_mat = np.meshgrid(np.arange(self.rows), np.arange(self.columns))
            xi_mat = np.reshape(xi_mat, (1, -1))
            yi_mat = np.reshape(yi_mat, (1, -1))
            zi_mat = np.reshape(np.ones((self.rows, self.columns), dtype=float), (1, -1))
            loci_mat = np.concatenate((xi_mat, yi_mat, zi_mat), axis=0)
            loc = vox2abs * loci_mat
            self.loc_x[:, :, z] = loc[0].reshape((self.rows, self.columns))
            self.loc_y[:, :, z] = loc[1].reshape((self.rows, self.columns))
            self.loc_z[:, :, z] = loc[2].reshape((self.rows, self.columns))

    def get_loc_matrix(self):
        if not self.loc_x:
            self.gen_loc_matrix()
        return self.loc_x, self.loc_y, self.loc_z

    def image_norm(self, low=-1024, high=2048, scale=255, do_round=True):
        self.image_3d_norm = (self.image_3d_raw - low) / (high - low)
        self.image_3d_norm[self.image_3d_norm < 0] = 0
        self.image_3d_norm[self.image_3d_norm > 1] = 1
        self.image_3d_norm *= scale
        if do_round:
            self.image_3d_norm = np.round(self.image_3d_norm)


class DicomRTst:
    def __init__(self, rtst_file):

        self.patient_id = ''
        self.rtst_file = rtst_file

        self.RTst = None
        self.poi_list = []
        self.poi_name_to_index = {}
        self.roi_list = []
        self.roi_name_to_index = {}
        self.rois_raw = []
        self.series_uid = ''
        self.file_uid = ''
        self.ref_series = ''
        self.roimask = {}
        self.roi_z_have_mask = {}

        # this part come from image
        self.dicom_img = None
        self.rows = 0
        self.columns = 0
        self.z_length = 0
        self.__uid2z = {}

        if rtst_file is None:
            logger.info('None file array')
        else:
            self.load_rtst(rtst_file)
            self.__read_roi_name_list()

    def load_rtst(self, file_name):
        logger.debug('Importing patient RTst. Please wait...')
        dp = DicomParser(filename=file_name)
        self.patient_id = dp.ds.PatientID
        logger.debug('patient ID: ' + self.patient_id)
        self.RTst = dp
        self.rois_raw = dp.get_structures()
        self.__read_base_info()

    def __read_base_info(self):
        series_info = self.RTst.get_series_info()
        self.series_uid = series_info['uid']
        self.file_uid = self.RTst.get_sop_instance_uid()
        self.ref_series = self.RTst.get_referenced_series()

    def read_ref_series(self, file_list):
        if isinstance(file_list, list):
            dicom_img = DicomImage(file_list)
            self.dicom_img = DicomImage(file_list)
            self.rows = dicom_img.rows
            self.columns = dicom_img.columns
            self.z_length = dicom_img.z_length
        else:
            logger.error('the reference image series ' + self.patient_id + ' is not a list!!!')
            logger.error('the file is ' + self.rtst_file)
            return 0

    def __read_roi_name_list(self):
        rois = self.rois_raw
        for key in rois:
            structure = rois[key]
            if structure['RTROIType'] == 'MARKER':
                self.poi_list.append(structure['name'])
                self.poi_name_to_index[structure['name']] = int(key)
            else:
                self.roi_list.append(structure['name'])
                self.roi_name_to_index[structure['name']] = int(key)

    def find_roi_index(self, roi_name):
        if roi_name in self.roi_name_to_index:
            return self.roi_name_to_index[roi_name]
        else:
            return 'Not find'

    def find_roi_index_capitalization_no_sensitive(self, roi_name):
        for key in self.roi_name_to_index:
            if roi_name == key.lower():
                return self.roi_name_to_index[key]
        return 'Not find'

    def _initial_3d_grid(self):
        return np.zeros((self.rows, self.columns, self.z_length), dtype=bool)

    def _get_z_by_image_file_uid(self, img_file_uid):
        return self.__uid2z[img_file_uid]

    '''
    def _initial_2d_grid(self, img_file_uid):
        # Found the image with RefImageUID
        z = self.dicom_img.get_z_by_image_file_uid(img_file_uid)
        x, y = self.dicom_img.get_img_mesh(z)
        x, y = x.flatten(), y.flatten()
        imagegridpoints = np.vstack((x, y)).T
        return imagegridpoints, z
    '''

    def create_3d_mask(self, roi_name, capitalization_sensitive):
        def _get_contours_from_plane(plane):
            # return contours
            contour_list = []
            for contour_i in plane:
                contour_data = contour_i['contourData']
                # Add  points to the list of contours
                contour_list.append(contour_data)
            return contour_list

        if capitalization_sensitive:
            roi_index = self.find_roi_index(roi_name)
        else:
            roi_index = self.find_roi_index_capitalization_no_sensitive(roi_name)
        # here check again
        if roi_index == 'Not find':
            logger.warning("provide zero grid for [" + roi_name + "] in " + self.patient_id)
            self.roimask[roi_name] = self._initial_3d_grid()
            return 0
        else:
            logger.debug("find ROI [" + roi_name + "] in " + self.patient_id)

        structure = self.rois_raw[roi_index]
        sPlanes = structure['planes']
        logger.debug("Calculating contour grid for " + roi_name)
        # Initial ROI 3d grid
        roigrid = self._initial_3d_grid()
        z_location_list = []
        for z in sPlanes:
            sPlane = sPlanes[z]
            # Get Ref image grid
            # logger.debug('Find the image plane for this contour at z = ' + str(z))
            imageUID = sPlane[0]['RefUID']
            if len(imageUID) == 0:
                # deal with this later
                pass
            else:
                # Get initial mask
                # image_grid_points, z_location = self._initial_2d_grid(imageUID)
                # logger.debug('Find the image grid location for this contour at z = ' +
                #      str(z_location))
                # z_location_list.append(z_location)

                z_location = self.dicom_img.get_z_by_image_file_uid(imageUID)

                # Get the contours
                contours = _get_contours_from_plane(sPlane)
                for contour in contours:
                    contour_rel = self.dicom_img.points_abs2rel(imageUID, contour)
                    contour_rel = np.round(np.transpose(contour_rel)[:, 0:2]).astype(np.int32)

                    '''
                    p = mpath.Path(contour)
                    # p.contains_points is the way to calculate ROI
                    grid = p.contains_points(image_grid_points)
                    grid = grid.reshape((self.rows, self.columns))
                    '''

                    grid = np.zeros((self.rows, self.columns), dtype=np.int8)
                    cv2.drawContours(image=grid, contours=[contour_rel],
                                     contourIdx=-1, color=255, thickness=cv2.FILLED)
                    grid = grid.astype(np.bool)

                    if np.amax(roigrid[:, :, z_location]) == 0:
                        roigrid[:, :, z_location] = grid
                    else:
                        # user XOR to combine mask
                        roigrid[:, :, z_location] = roigrid[:, :, z_location] ^ grid

        self.roimask[roi_name] = roigrid
        self.roi_z_have_mask[roi_name] = np.unique(np.array(z_location_list))
        return True


class DicomDose:
    def __init__(self, dose_file):

        self.patient_id = ''
        self.dose_uid = ''
        self.dose_dp = None
        self.dose_arary = None
        self.dose_info = None
        self.dose_interpolated = None
        self.dicom_rtst = []
        self.dicom_img = None
        self.dose_dim = None

        self.ref_study = ''
        self._affine_mat = None

        self.loc_x = None
        self.loc_y = None
        self.loc_z = None
        self.loc = None
        try:
            self.load_dose(dose_file)
        except:
            raise

    def load_dose(self, file_name):
        logger.debug('Importing patient dose. Please wait...')
        dp = DicomParser(filename=file_name)
        self.patient_id = dp.ds.PatientID
        logger.debug('patient ID: ' + self.patient_id)
        self.dose_dp = dp
        self._read_base_info()
        self.read_dose_array()

    def read_dose_array(self):
        self.dose_arary = self.dose_dp.ds.pixel_array * self.dose_info['dosegridscaling']
        self.dose_arary = np.transpose(self.dose_arary, (1, 2, 0))
        self.dose_dim = np.shape(self.dose_arary)

    def _read_base_info(self):
        self.dose_uid = self.dose_dp.ds.SOPInstanceUID
        self.ref_study = self.dose_dp.ds.StudyInstanceUID
        self.dose_info = self.dose_dp.get_dose_info()

    def read_ref_series(self, file_list):
        self.dicom_img = DicomImage(file_list)

    def get_affine_mat(self):
        return AffineMatrix(self.dose_dp.ds)

    def __is_simple_case(self):
        ds = self.dose_dp.ds
        image_orientation = ds.ImageOrientationPatient
        F11 = image_orientation[0]
        F21 = image_orientation[1]
        F31 = image_orientation[2]
        F12 = image_orientation[3]
        F22 = image_orientation[4]
        F32 = image_orientation[5]
        if F11 == 1 and F21 == 0 and F31 == 0 \
                and F12 == 0 and F22 == 1 and F32 == 0:
            return True
        else:
            return False

    def gen_loc_vector(self):
        ds = self.dose_dp.ds
        delta_r = ds.PixelSpacing[0]
        delta_c = ds.PixelSpacing[1]

        T11 = ds.ImagePositionPatient[0]
        T12 = ds.ImagePositionPatient[1]
        T13 = ds.ImagePositionPatient[2]

        image_orientation = ds.ImageOrientationPatient
        F11 = image_orientation[0]
        F21 = image_orientation[1]
        F31 = image_orientation[2]
        F12 = image_orientation[3]
        F22 = image_orientation[4]
        F32 = image_orientation[5]

        # check again
        assert F11 == 1
        assert F21 == 0
        assert F31 == 0
        assert F12 == 0
        assert F22 == 1
        assert F32 == 0

        self.loc_x = np.arange(self.dose_dim[0]) * delta_r + T12
        self.loc_y = np.arange(self.dose_dim[1]) * delta_c + T11
        self.loc_z = np.array(ds.GridFrameOffsetVector) + T13

    def gen_loc_matrix(self):
        ds = self.dose_dp.ds
        self.loc_x = np.arange(self.dose_dim[0])
        self.loc_y = np.arange(self.dose_dim[1])
        self.loc_z = np.array(ds.GridFrameOffsetVector)

    # TODO: it have some strange bug !!! need time to figure out
    def interp_dose(self):
        self.dicom_img.gen_loc_matrix()
        if self.__is_simple_case():
            self.gen_loc_vector()
            xi_mat = np.reshape(self.dicom_img.loc_x, (-1, 1))
            yi_mat = np.reshape(self.dicom_img.loc_y, (-1, 1))
            zi_mat = np.reshape(self.dicom_img.loc_z, (-1, 1))
            loci_mat = np.concatenate((yi_mat, xi_mat, zi_mat), axis=1)
            dose_array_img_trans = interpn(
                points=[self.loc_x, self.loc_y, self.loc_z],
                values=self.dose_arary,
                bounds_error=False,
                fill_value=0.0,
                xi=loci_mat)
            self.dose_interpolated = dose_array_img_trans.reshape([self.dicom_img.rows,
                                                                   self.dicom_img.columns,
                                                                   self.dicom_img.z_length])
        else:
            # need transfer to dose domain to fix this problem
            self.gen_loc_matrix()
            xi_mat = np.reshape(self.dicom_img.loc_x, (1, -1))
            yi_mat = np.reshape(self.dicom_img.loc_y, (1, -1))
            zi_mat = np.reshape(self.dicom_img.loc_z, (1, -1))
            ii_mat = np.reshape(np.ones(self.dicom_img.loc_x.shape), (1, -1))
            loci_mat = np.concatenate((xi_mat, yi_mat, zi_mat, ii_mat), axis=0)
            affine_mat = self.get_affine_mat()
            dose_abs2vox = affine_mat.abs2vox_4D
            dose_loci_mat = dose_abs2vox * loci_mat
            dose_loci_mat = dose_loci_mat[0:3, :]
            dose_loci_mat = np.array(np.transpose(dose_loci_mat, (1, 0)))
            dose_array_img_trans = interpn(
                points=[self.loc_y, self.loc_x, self.loc_z],
                values=np.transpose(self.dose_arary, (1, 0, 2)),
                bounds_error=False,
                fill_value=0.0,
                xi=dose_loci_mat)
            self.dose_interpolated = dose_array_img_trans.reshape([self.dicom_img.rows,
                                                                   self.dicom_img.columns,
                                                                   self.dicom_img.z_length])
        return self.dose_interpolated


"""
def dicom_test():
    # 1. test scan function
    # 2. test print tree function
    # 3. test find all dose uid and dose files
    # 4. test find image file list by dose uid
    # 5. test find rtst file by dose uid
    # 6. test find image file list by rtst

    # test 1
    dicomdir = DicomDirectory('C:/data/Desktop/dose')

    # test 2
    dicomdir.print_tree('tree.txt')

    # test 3
    all_RTdose_uid = dicomdir.get_all_dose_uid()
    all_RTdose_files = dicomdir.get_all_dose_file()
    print(all_RTdose_uid)
    print(all_RTdose_files)

    for RTdose_uid, RTdose_file in zip(all_RTdose_uid, all_RTdose_files):
        # read dose file
        dicom_dose = DicomDose(RTdose_file)

        # test 4
        img_file_list = dicomdir.get_ref_img_files_by_dose_uid(RTdose_uid)
        if img_file_list == 0:
            logger.warning('Not find image files for this RTdose: ' + RTdose_file)
            logger.warning('This RTdose belong to patient: ' + dicom_dose.patient_id)
            continue

        # test 5
        rtst_file = dicomdir.get_rtst_file_by_dose_uid(RTdose_uid)
        if rtst_file == 0:
            logger.warning('Not find rtst file for this RTdose: ' + RTdose_file)
            logger.warning('This RTdose belong to patient: ' + dicom_dose.patient_id)
            continue

        dicom_rtst = DicomRTst(rtst_file)
        rtst_file_uid = dicom_rtst.file_uid
        # test 6
        rtst_img_file_list = dicomdir.get_ref_img_files_by_rtst_file_uid(rtst_file_uid)
        if img_file_list == 0:
            logger.warning('Not find image files for this rtst: ' + rtst_file)
            logger.warning('This RTdose belong to patient: ' + rtst_file.patient_id)
            continue

        dicom_rtst.read_ref_series(rtst_img_file_list)
        dicom_dose.read_ref_series(img_file_list)
        roi_list = ['LS']
        for roi in roi_list:
            if dicom_rtst.find_roi_index(roi) == 'Not find':
                continue
            dicom_rtst.create_3d_mask(roi)

        dose_array_img_trans = dicom_dose.interp_dose()
        slice_ind = 28
        # print(dose_array_img_trans[:, :, slice_ind])
        dicom_dose.dicom_img.image_norm(low=-1024, high=2048, scale=255, do_round=True)
        ct_img = cv2.convertScaleAbs(dicom_dose.dicom_img.image_3d_norm[:, :, slice_ind])
        ct_img = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2RGB)
        if dose_array_img_trans[:, :, slice_ind].max() == 0:
            dose_img = cv2.convertScaleAbs(
                dose_array_img_trans[:, :, slice_ind] * 255)
        else:
            dose_img = cv2.convertScaleAbs(
                dose_array_img_trans[:, :, slice_ind] /
                dose_array_img_trans[:, :, slice_ind].max() * 255)
        dose_img = cv2.applyColorMap(dose_img, cv2.COLORMAP_JET)
        ptv_mask = dicom_rtst.roimask['LS'][:, :, slice_ind]
        gray = cv2.convertScaleAbs(ptv_mask * 255)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_img = cv2.addWeighted(ct_img, 1.0, dose_img, 0.5, 0)
        cv2.drawContours(final_img, contours, -1, (0, 255, 0), 3)
        cv2.imshow('1', final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

# dicom_test()
