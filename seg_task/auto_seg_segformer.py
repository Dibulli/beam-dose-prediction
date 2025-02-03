import logging

import numpy as np
import torch

from lib.color_palette import *
from lib.dicom.dicom_directory import DicomDirectory
from lib.dicom.dicom_image_series import DicomCTSeries
from lib.dicom.dicom_dose_series import DicomDoseSeries
from lib.dicom.dicom_rtst import masks_to_dicomrt
from lib.dicom.dicom_rtst import DicomRTST
from lib.network.segformer_pytorch import Segformer
from lib.network.unet import UNet
from lib.utilities import shape_check_and_rescale
from lib.utilities import to_abs_path
# from seg_task.Beams_Contour_Generator import BeamContourGeneration
from Beams_Contour_Generator import BeamContourGeneration
import einops

import os

logger = logging.getLogger(__name__)


def run_auto_seg(cfg, dcm_img_folder=None, output_folder=None):
    if dcm_img_folder is None:
        dcm_img_folder = cfg.predict_dicom_path
    if output_folder is None:
        output_folder = cfg.dose_output_path

    dcm_img_folder = to_abs_path(dcm_img_folder)
    output_folder = to_abs_path(output_folder)
    model_file_path = to_abs_path(cfg.model_file_path)
    print("output_folder is:[{:}]".format(output_folder))
    print("using [{:}] model pth file;".format(model_file_path))
    print("scaning dicom folder is [{:}]".format(dcm_img_folder))
    # set device before build
    # modelA = Segformer()
    modelA = UNet(n_channels=6, n_classes=1)

    check_point = torch.load(model_file_path)
    # model.load_state_dict(torch.load(model_file_path, map_location='cpu'))
    modelA.load_state_dict(check_point['modelA'])

    if cfg.cpu_or_gpu == 'gpu':
        gpu = True
        device = torch.device('cuda:' + str(cfg.gpu_id))
        modelA.to(device)

    # model.eval()
    modelA.train()

    logger.info('start auto segmentation.')
    logger.info('start searching series')

    for root, dirs, files in os.walk(dcm_img_folder):
        for dir in dirs:
            real_path = root + dir + os.sep
        # real_path = root

            pred_dicomdir = DicomDirectory(real_path)
            pred_dicomdir.scan()

            for dicom_mr_series_base_info in pred_dicomdir.series_iter(series_type='CT Image Storage'):

                for dicom_rtst_series in pred_dicomdir.series_iter(series_type='RT Structure Set Storage'):

                    dicom_mr_series = DicomCTSeries(dicom_mr_series_base_info)
                    dicom_mr_series.load_data()
                    # debug

                    rtst_file = dicom_rtst_series.single_file
                    rtst_dataset = DicomRTST(rtst_file)
                    pt_id = rtst_dataset.PatientID

                    # normalization setting
                    if cfg.norm_method == 'fix':
                        dicom_mr_series.norm_method = 'fix'
                        dicom_mr_series.min_value = cfg.norm_low
                        dicom_mr_series.max_value = cfg.norm_high

                    slice_img = dicom_mr_series.normalized_voxel_array
                    slice_img = shape_check_and_rescale(slice_img, cfg.dim_y, cfg.dim_x)
                    # half_dim = int((cfg.channels_of_input_images - 1) / 2)
                    z_dim = slice_img.shape[2]
                    imgs = einops.repeat(slice_img, "h w (b z) -> (repeat b) h w z", repeat=9, b=1)
                    # imgs = einops.rearrange(slice_img, "h w (b z) -> b h w z", b=1)

                    # imgs = np.expand_dims(slice_img, axis=0)
                    # [b, h, w, z]
                    # concatenate mask and img, put them together into model.
                    roi_dicts = cfg.roi_dicts
                    slice_mask = np.zeros(list(slice_img.shape) + [len(roi_dicts)], dtype=int)

                    all_roi_exist = rtst_dataset.roi_exist_check_and_match(roi_dicts)
                    if (not all_roi_exist):
                        logger.warning('[{}] do not have roi[{}].'.format(pt_id, roi_dicts))

                    for roi_ind_in_conf, single_roi_conf in enumerate(roi_dicts):
                        roi_mask, success_flag = rtst_dataset.create_3d_mask(roi_ind_in_conf, dicom_mr_series)
                        roi_mask = shape_check_and_rescale(roi_mask, cfg.dim_y, cfg.dim_x)

                        slice_mask[:, :, :, roi_ind_in_conf] = roi_mask

                    bc = BeamContourGeneration(slice_mask, patient_id=None)
                    # slice_beam = bc.output()
                    beams = bc.output()
                    beams = np.array(beams)
                    # [512, 512, z, 9, 3]
                    # body = slice_mask[:, :, :, 0]
                    dose_pred = np.zeros((9, 1, cfg.dim_y, cfg.dim_x, z_dim), dtype=np.float32)
                    # dose_pred = np.zeros((1, 1, cfg.dim_y, cfg.dim_x, z_dim), dtype=np.float32)

                    # slice_mask[:, :, :, -1] = slice_beam
                    #
                    # print("slice_mask.shape[{}]".format(slice_mask.shape))
                    with torch.no_grad():
                        for z in range(z_dim):
                            # print("starting predict dose in layer [{:}]".format(z))
                            img = imgs[:, :, :, z]
                            img = einops.repeat(img, "(b c) h w -> b c h w", c=1)


                            # contours = slice_beam[:, :, z, :]
                            contours = beams[:, :, :, z, :]

                            # contours = slice_beam[:, :, z]
                            # [512, 512, 9, 3]
                            # contours = einops.rearrange(contours, "h w (b c) -> b c h w", b=9)
                            contours = einops.rearrange(contours, "c h w b -> b c h w")

                            # contours = einops.rearrange(contours, "h w (b c) -> b c h w", b=1)

                            # masks = slice_mask[:, :, z, :]
                            # masks = einops.repeat(masks, 'h w (b c) -> (repeat b) c h w', repeat=9, b=1)
                            # body = masks[:, :1, :, :]
                            # ptvs = masks[:, 1:4, :, :]
                            # oars = np.sum(masks[:, 4:, :, :], axis=1, keepdims=True)
                            # reshape_masks = np.concatenate((body, ptvs, oars), axis=1)

                            # input_ib = np.concatenate((img, contours, reshape_masks), axis=1)
                            input_ib = np.concatenate((img, contours), axis=1)



                            input_ib = torch.from_numpy(input_ib)
                            input_ib = input_ib.type(torch.FloatTensor)
                            # masks = torch.from_numpy(masks)
                            # masks = masks.type(torch.FloatTensor)

                            if gpu:
                                input_ib = input_ib.to(device)
                                # masks = masks.to(device)

                                pred = modelA(input_ib)
                                dose_pred[:, :, :, :, z]  = pred.cpu().numpy()

                        total_pred = np.sum(dose_pred, axis=0, keepdims=False)
                        reshaped_pred = einops.rearrange(dose_pred, 'b c h w z -> h w z (b c)')
                        total_reshape = einops.rearrange(total_pred, 'c h w z -> h w z c')
                        pred_all = np.concatenate((reshaped_pred, total_reshape), axis=-1)
                    logger.info('dose autoseg finished')
                    np.save(output_folder + os.sep + str(pt_id) + "_-1.npy", pred_all)
                    # np.save(output_folder + os.sep + str(pt_id) + "_-1.npy", dose_pred)

                    # for i in range(9):
                    #     np.save(output_folder + os.sep + str(pt_id) + "_" + str(i) + ".npy", dose_pred[i, :, :, :])

                    logger.info('dose save completed')

                # masks_to_dicomrt(dicom_mr_series,
                #                  cfg.roi_dicts,
                #                  masks=masks_pred,
                #                  roi_color_set=set1,
                #                  output_folder=output_folder)

