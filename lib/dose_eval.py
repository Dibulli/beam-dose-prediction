import logging
import torch
import cv2
import os
import numpy as np
from scipy.ndimage import zoom
from torchsummary import summary
from torch.utils.data import DataLoader
from seg_task.data_statistics.useless_code_temp.seg_data import SegData
from lib.utilities import to_abs_path
from lib.utilities import clean_folder
from .pinnacle_file_helper import write_pinnacle_script
from .pinnacle_file_helper import write_pinnacle_roi
from .pinnacle_file_helper import read_pinnacle_img
from .seg_pef_eval import calc_performance
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def dice_cal_fun(channel, eps=1):
    def sub_dice(input, target):
        iflat = input[:, channel, :, :].reshape(-1, 1)
        tflat = target[:, channel, :, :].reshape(-1, 1)
        intersection = (iflat * tflat).sum()
        return (2. * intersection + eps) / (iflat.sum() + tflat.sum() + eps)
    return sub_dice


def dvh_cal(roi_mask, dose_matrix):
    # fig, ax = plt.subplots()
    dose_array_img_trans = dose_matrix
    dose_value = dose_array_img_trans[roi_mask]

    dose_bins = np.arange(0, 60, 0.1)
    dvh_diff_density, edges = np.histogram(dose_value, dose_bins, density=True)
    dvh_cum_density = np.cumsum(dvh_diff_density[::-1])[::-1]

    dvh = dvh_cum_density
    return dvh


class dose_eval():
    def __init__(self, cfg):
        self.cfg = cfg
        self.channel_info = {'ptvap704': 0, 'ptvap660': 1, 'ptvap608': 2, 'ptvap600': 3,
                             'ptvap560': 4, 'ptvap540': 5}

        self.test_ds = SegData(self.cfg, 'test')
        self.test_loader = DataLoader(self.test_ds, batch_size=self.cfg.batch_size,
                                      num_workers=self.cfg.train_loader_works,
                                      pin_memory=True)
        self.model_file_path = to_abs_path(cfg.model_file_path)
        if cfg.cpu_or_gpu == 'gpu':
            self.gpu = True
            self.device = torch.device('cuda:' + str(cfg.gpu_id))
        else:
            self.gpu = False
            self.device = 'cpu'

        self.model = None
        self._model_set()


    def _model_build(self):
        if 'model_type' in self.cfg:
            if self.cfg.model_type == 'unet_res':
                self.model_type = 'unet_res'
                from lib.network.unet_res import UNet
            elif self.cfg.model_type == 'unet_res_big':
                self.model_type = 'unet_res_big'
                from lib.network.unet_res_big import UNet
            elif self.cfg.model_type == 'unet_mask':
                self.model_type = 'unet_mask'
                from lib.network.unet_mask import UNet
            elif self.cfg.model_type == 'unet_attention':
                self.model_type = 'unet_attention'
                from lib.network.unet_attention import UNet
            elif self.cfg.model_type == 'unet_type2':
                self.model_type = 'unet_type2'
                from lib.network.unet_type2 import UNet
            elif self.cfg.model_type == 'unet_deeper':
                self.model_type = 'unet_deeper'
                from lib.network.unet_deeper import UNet
            elif self.cfg.model_type == 'wnet':
                self.model_type = 'wnet'
                from lib.network.wnet import Wnet as UNet
            else:
                self.model_type = 'unet'
                from lib.network.unet import UNet
        else:
            self.model_type = 'unet'
            from lib.network.unet import UNet

        self.model = UNet(self.cfg.channels_of_input_images, self.cfg.channels_of_output_mask, 1)
        summary(self.model.to(self.device), (6, 512, 512))

        # set device before build
        if self.gpu:
            self.model.to(self.device)

    def _load_weight(self):
        logger.info('load weight')
        print("Load model from: %s" % self.model_file_path)
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))
        logger.info('load weight successful')

    def _model_set(self):
        self._model_build()
        self._load_weight()

    def dose_eval(self):
        set1 = [[255, 127, 53],
                [84, 138, 255],
                [99, 169, 0],
                [178, 67, 201],
                [36, 136, 0],
                [219, 26, 156]],

        # set model to evaluation
        self.model.eval()
        check_img_path = to_abs_path(self.cfg.test_out_path)
        clean_folder(check_img_path)
        # loop batch
        true_dose_batch = []
        true_ptv_batch = []
        predicted_batch =[]
        ct_img_batch = []
        roi_batch = {'ptvap704': [], 'ptvap660': [], 'ptvap608': [], 'ptvap600': [], 'ptvap560': [], 'ptvap540': []}
        with torch.no_grad():
            for batch_i, batch_data in enumerate(self.test_loader):

                batch_img = batch_data[0].numpy()
                batch_mask = batch_data[1].numpy()
                batch_pt_id = batch_data[3]
                batch_z = batch_data[4].numpy()
                batch_dose = batch_data[2].numpy()
                if self.gpu:
                    batch_img_gpu = batch_data[0].to(self.device)
                pred = self.model(batch_img_gpu)
                dose_pred = pred[1].cpu().numpy()

                for sample_i in range(len(batch_pt_id)):
                    ptv_channel = self.channel_info['PTV']
                    true_dose_batch.append(batch_dose[sample_i, 0, :, :])
                    true_ptv_batch.append(batch_mask[sample_i, ptv_channel, :, :])
                    predicted_batch.append(dose_pred[sample_i, 0])
                    ct_img = np.array(batch_img[sample_i, round(self.cfg.channels_of_input_images / 2), :, :] * 255,
                                      dtype=np.uint8)
                    ct_img = ct_img.reshape((ct_img.shape[0], ct_img.shape[1], 1))
                    ct_img_rgb = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2RGB)
                    ct_img_batch.append(ct_img_rgb)
                    for roi in self.roi_list:
                        roi_channel = self.channel_info[roi]
                        roi_batch[roi].append(batch_mask[sample_i, roi_channel, :, :])


        true_ptv_batch = np.array(true_ptv_batch)
        predicted_batch = np.array(predicted_batch)
        true_dose_batch = np.array(true_dose_batch)
        ct_img_batch = np.array(ct_img_batch)
        for roi in roi_batch.keys():
            roi_mask = np.array(roi_batch[roi], dtype=bool).transpose((1, 2, 0))
            fig, ax = plt.subplots()
            dose_bins = np.arange(0, 59.9, 0.1)
            for mode in ['predicted', 'truth']:
                if mode == 'truth':
                    roi_dvh = dvh_cal(roi_mask, true_dose_batch.transpose((1, 2, 0)) * 100) / 10.0
                else:
                    roi_dvh = dvh_cal(roi_mask, predicted_batch.transpose((1, 2, 0)) * 100) / 10.0
                # Plot dvh
                color = 'blue' if mode == 'predicted' else 'red'
                plt.plot(dose_bins, roi_dvh, color=color, label=mode)
                ax.legend()
                # plt.hist(roi_dvh, bins=100, color=color, histtype='stepfilled')
            file_name = 'DVH_%s' % roi + '.png'
            dvh_out_path = os.path.join(check_img_path, file_name)
            title = 'DVH_%s' % roi
            plt.title(title)
            plt.savefig(dvh_out_path)

        hd = {}
        hd95 = {}
        ahd = {}
        dice = {}
        jaccard = {}
        # print(true_ptv_batch.shape, predicted_batch.shape, true_dose_batch.shape)
        true_ptv = true_ptv_batch.transpose((1, 2, 0))
        predicted_batch_dose_mask = (predicted_batch >= 0.5).transpose((1, 2, 0))
        predicted_batch_dose95_mask = (predicted_batch >= 0.45).transpose((1, 2, 0))
        true_dose_mask = (true_dose_batch >= 0.5).transpose((1, 2, 0))

        for i in range(predicted_batch_dose_mask.shape[-1]):
            out_sample_t = np.array(true_ptv[:, :, i: i + 1] * 255,
                                    dtype=np.uint8)
            ret_t, binary_t = cv2.threshold(out_sample_t, 127, 255, cv2.THRESH_BINARY)
            contours_t, hierarchy_t = cv2.findContours(binary_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            out_sample_p = np.array(predicted_batch_dose_mask[:, :, i: i + 1] * 255,
                                    dtype=np.uint8)
            ret_p, binary_p = cv2.threshold(out_sample_p, 127, 255, cv2.THRESH_BINARY)
            contours_p, hierarchy_p = cv2.findContours(binary_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            out_sample_d = np.array(true_dose_mask[:, :, i: i + 1] * 255,
                                    dtype=np.uint8)
            ret_d, binary_d = cv2.threshold(out_sample_d, 127, 255, cv2.THRESH_BINARY)
            contours_d, hierarchy_d = cv2.findContours(binary_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            ptv_t = np.zeros((out_sample_t.shape[0], out_sample_t.shape[0], 3))
            ptv_t[:, :, :] = ct_img_batch[i]
            cv2.drawContours(ptv_t, contours_t, -1, set1[1], 2)
            cv2.drawContours(ptv_t, contours_d, -1, set1[2], 2)
            ptv_p = np.zeros((out_sample_t.shape[0], out_sample_t.shape[0], 3))
            ptv_p[:, :, :] = ct_img_batch[i]
            cv2.drawContours(ptv_p, contours_t, -1, set1[1], 2)
            cv2.drawContours(ptv_p, contours_p, -1, set1[2], 2)
            # file_name_t = 'dose_conform_truth' + '_slice_%s' % i + '.png'
            # file_name_p = 'dose_conform_predicted' + '_slice_%s' % i + '.png'
            file_name_c = 'dose_conform_combined' + '_slice_%s' % i + '.png'
            # conform_t_out_path = os.path.join(check_img_path, file_name_t)
            # conform_p_out_path = os.path.join(check_img_path, file_name_p)
            conform_c_out_path = os.path.join(check_img_path, file_name_c)
            combined_conform = np.zeros((ptv_t.shape[0], 2 * ptv_t.shape[1], 3))
            combined_conform[:, 0: ptv_t.shape[0], :] = ptv_t
            combined_conform[:, ptv_t.shape[0]:, :] = ptv_p
            cv2.imwrite(conform_c_out_path, combined_conform)
            # cv2.imwrite(conform_t_out_path, ptv_t)
            # cv2.imwrite(conform_p_out_path, ptv_p)

        # print(true_dose_mask[255, :, 30])
        # Data check
        # for i in range(predicted_batch_dose_mask.shape[-1]):
        #     file_name = 'dose' + '_%s' % i + '.png'
        #     test_out_path = os.path.join(check_img_path, file_name)
        #     out_sample_t = np.array(true_ptv[:, :, i: i + 1] * 255.0,
        #                             dtype=np.uint8)
        #     out_sample_p = np.array(predicted_batch_dose_mask[:, :, i: i + 1] * 255.0,
        #                             dtype=np.uint8)
        #     out_sample_d = np.array(true_dose_mask[:, :, i: i + 1] * 255.0,
        #                             dtype=np.uint8)
        #     out_sample = np.zeros(
        #         (out_sample_t.shape[0] + out_sample_t.shape[0] + out_sample_t.shape[0], out_sample_t.shape[1], 1))
        #     out_sample[0: out_sample_t.shape[0]] = out_sample_t
        #     out_sample[out_sample_t.shape[0]:out_sample_t.shape[0] + out_sample_p.shape[0]] = out_sample_p
        #     out_sample[out_sample_t.shape[0] + out_sample_p.shape[0]:] = out_sample_d
        #     cv2.imwrite(test_out_path, out_sample)
        for mode in ['predicted', 'truth']:
            if mode == 'predicted':
                metric_1, metric_2, metric_3, metric_4, metric_5 = calc_performance(true_ptv, predicted_batch_dose_mask)
            else:
                metric_1, metric_2, metric_3, metric_4, metric_5 = calc_performance(true_ptv, true_dose_mask)
            hd[mode] = metric_1
            hd95[mode] = metric_2
            ahd[mode] = metric_3
            dice[mode] = metric_4
            jaccard[mode] = metric_5
        print('hd: %s \n' % hd, 'hd95: %s \n' % hd95, 'ahd: %s \n' % ahd, 'dice: %s \n' % dice, 'jaccard: %s \n' % jaccard)


    def roi_eval(self):
        # set model to evaluation
        self.model.eval()
        check_img_path = to_abs_path(self.cfg.test_out_path)
        clean_folder(check_img_path)
        # loop batch
        true_batch = []
        predicted_batch =[]
        with torch.no_grad():
            for batch_i, batch_data in enumerate(self.test_loader):

                batch_img = batch_data[0].numpy()
                batch_mask = batch_data[1].numpy()
                batch_pt_id = batch_data[3]
                batch_z = batch_data[4].numpy()
                batch_dose = batch_data[2].numpy()
                if self.gpu:
                    batch_img_gpu = batch_data[0].to(self.device)
                pred = self.model(batch_img_gpu)
                dose_pred = pred[1].cpu().numpy()
                mask_pred = pred[0].cpu().numpy()

                for sample_i in range(len(batch_pt_id)):
                    true_batch.append(batch_mask[sample_i])
                    predicted_batch.append(mask_pred[sample_i])

        true_batch = np.array(true_batch)
        predicted_batch = np.array(predicted_batch)

        hd = {}
        hd95 = {}
        ahd = {}
        dice = {}
        jaccard = {}
        for roi in self.roi_list:
            mask_channel = self.channel_info[roi]
            true_batch_roi = true_batch[:, mask_channel, :, :].transpose((1, 2, 0))
            predicted_batch_roi = (predicted_batch[:, mask_channel, :, :] > 0.5).transpose((1, 2, 0))
            # Data check
            # for i in range(predicted_batch_roi.shape[-1]):
            #     file_name = '%s' % roi + '_%s' % i + '.png'
            #     test_out_path = os.path.join(check_img_path, file_name)
            #     out_sample_t = np.array(true_batch_roi[:, :, i: i + 1] * 255.0,
            #                                   dtype=np.uint8)
            #     out_sample_p = np.array(predicted_batch_roi[:, :, i: i + 1] * 255.0,
            #                                   dtype=np.uint8)
            #     out_sample = np.zeros((out_sample_t.shape[0] + out_sample_t.shape[0], out_sample_t.shape[1], 1))
            #     out_sample[0: out_sample_t.shape[0]] = out_sample_t
            #     out_sample[out_sample_t.shape[0]:] = out_sample_p
            #     cv2.imwrite(test_out_path, out_sample)
            metric_1, metric_2, metric_3, metric_4, metric_5 = calc_performance(true_batch_roi, predicted_batch_roi)
            hd[roi] = metric_1
            hd95[roi] = metric_2
            ahd[roi] = metric_3
            dice[roi] = metric_4
            jaccard[roi] = metric_5
        print('hd: %s \n' % hd, 'hd95: %s \n' % hd95, 'ahd: %s \n' % ahd, 'dice: %s \n' % dice, 'jaccard: %s \n' % jaccard)

    def dose_show(self):
        # set model to evaluation
        self.model.eval()
        check_img_path = to_abs_path(self.cfg.test_out_path)

        set1 = [[255, 127, 53],
                [84, 138, 255],
                [99, 169, 0],
                [178, 67, 201],
                [36, 136, 0]]

        clean_folder(check_img_path)

        # loop batch
        with torch.no_grad():
            for batch_i, batch_data in enumerate(self.test_loader):

                batch_img = batch_data[0].numpy()
                batch_mask = batch_data[1].numpy()
                batch_pt_id = batch_data[3]
                batch_z = batch_data[4].numpy()
                batch_dose = batch_data[2].numpy()
                if self.gpu:
                    batch_img_gpu = batch_data[0].to(self.device)
                pred = self.model(batch_img_gpu)
                dose_pred = pred[1].cpu().numpy()
                mask_pred = pred[0].cpu().numpy()

                for sample_i in range(len(batch_pt_id)):
                    logger.info('batch [{}] sample [{}]'.format(
                        str(batch_i), str(sample_i)))
                    pt_id = batch_pt_id[sample_i]

                    combine_check_img_rgb = None
                    combine_pred_img_rgb = None
                    # check_img = np.array(batch_img[sample_i, :, :, :].transpose(1, 2, 0) * 255,
                    #                      dtype=np.uint8)
                    check_img = np.array(batch_img[sample_i, round(self.cfg.channels_of_input_images / 2), :, :] * 255,
                                         dtype=np.uint8)
                    check_img = check_img.reshape((check_img.shape[0], check_img.shape[1], 1))
                    check_img_rgb = cv2.cvtColor(check_img, cv2.COLOR_GRAY2RGB)
                    check_dose = np.array(batch_dose[sample_i, :, :, :].transpose(1, 2, 0) * 255,
                                          dtype=np.uint8)
                    check_pred_dose = np.array(dose_pred[sample_i, :, :, :].transpose(1, 2, 0) * 255,
                                          dtype=np.uint8)
                    check_dose_rgb = cv2.applyColorMap(check_dose, cv2.COLORMAP_JET)
                    check_pred_dose_rgb = cv2.applyColorMap(check_pred_dose, cv2.COLORMAP_JET)

                    for mask_channel in range(5):
                        roi_color = set1[mask_channel]
                        roi_check_img = batch_mask[sample_i, mask_channel, :, :]

                        roi_check_img_rgb = np.repeat(roi_check_img[:, :, np.newaxis], 3, axis=2)
                        roi_check_img_rgb = roi_check_img_rgb * np.array(roi_color, dtype=int)
                        roi_check_img_rgb = np.array(roi_check_img_rgb, dtype=np.uint8)

                        if combine_check_img_rgb is None:
                            combine_check_img_rgb = cv2.addWeighted(check_img_rgb, 1.0, roi_check_img_rgb,
                                                                    0.7, 0)
                        else:
                            combine_check_img_rgb = cv2.addWeighted(combine_check_img_rgb, 1.0, roi_check_img_rgb,
                                                                    0.7, 0)
                        pred_roi_color = set1[mask_channel]
                        pred_roi_check_img = (mask_pred[sample_i, mask_channel, :, :] > 0.3)
                        dice = np.zeros((len(batch_pt_id)))
                        dice[sample_i] = dice_cal_fun(channel=mask_channel)(batch_data[1][sample_i: sample_i + 1].cpu(),
                                                                             pred[0][sample_i: sample_i + 1].cpu()).item()

                        pred_roi_check_img_rgb = np.repeat(pred_roi_check_img[:, :, np.newaxis], 3, axis=2)
                        pred_roi_check_img_rgb = pred_roi_check_img_rgb * np.array(pred_roi_color, dtype=int)
                        pred_roi_check_img_rgb = np.array(pred_roi_check_img_rgb, dtype=np.uint8)

                        if combine_pred_img_rgb is None:
                            combine_pred_img_rgb = cv2.addWeighted(check_img_rgb, 1.0, pred_roi_check_img_rgb,
                                                                    0.7, 0)
                        else:
                            combine_pred_img_rgb = cv2.addWeighted(combine_pred_img_rgb, 1.0, pred_roi_check_img_rgb,
                                                                    0.7, 0)

                    combine_out = np.zeros((4 * combine_check_img_rgb.shape[0], combine_check_img_rgb.shape[1], 3))
                    combine_out[0: combine_check_img_rgb.shape[0], :, :] = combine_check_img_rgb
                    combine_out[combine_check_img_rgb.shape[0]: 2 * combine_check_img_rgb.shape[0], :, :] = combine_pred_img_rgb
                    combine_out[2 * combine_check_img_rgb.shape[0]: 3 * combine_check_img_rgb.shape[0], :, :] = check_dose_rgb
                    combine_out[3 * combine_check_img_rgb.shape[0]:, :, :] = check_pred_dose_rgb
                    dice_i = str(dice[sample_i])
                    file_name = '[{0:s}]_b[{1:s}]_s[{2:s}]_dice[{3:s}]img.png'.format(
                        pt_id, str(batch_i), str(sample_i), dice_i)
                    file_path = check_img_path + os.sep + file_name
                    cv2.imwrite(file_path, combine_out)


    def roi_show(self):
        # set model to evaluation
        self.model.eval()
        check_img_path = to_abs_path(self.cfg.test_out_path)

        set1 = [[255, 127, 53],
                [84, 138, 255],
                [99, 169, 0],
                [178, 67, 201],
                [36, 136, 0]]

        clean_folder(check_img_path)

        # loop batch
        with torch.no_grad():
            for batch_i, batch_data in enumerate(self.test_loader):

                batch_img = batch_data[0].numpy()
                batch_mask = batch_data[1].numpy()
                batch_pt_id = batch_data[2]
                batch_z = batch_data[3].numpy()
                if self.gpu:
                    batch_img_gpu = batch_data[0].to(self.device)
                pred = self.model(batch_img_gpu)
                mask_pred = pred.cpu().numpy()

                for sample_i in range(len(batch_pt_id)):
                    logger.info('batch [{}] sample [{}]'.format(
                        str(batch_i), str(sample_i)))
                    pt_id = batch_pt_id[sample_i]

                    combine_check_img_rgb = None
                    combine_pred_img_rgb = None
                    check_img = np.array(batch_img[sample_i, round(self.cfg.channels_of_input_images / 2), :, :] * 255,
                                         dtype=np.uint8)
                    check_img_rgb = cv2.cvtColor(check_img, cv2.COLOR_GRAY2RGB)

                    for mask_channel in range(5):
                        roi_color = set1[mask_channel]
                        roi_check_img = batch_mask[sample_i, mask_channel, :, :]

                        roi_check_img_rgb = np.repeat(roi_check_img[:, :, np.newaxis], 3, axis=2)
                        roi_check_img_rgb = roi_check_img_rgb * np.array(roi_color, dtype=int)
                        roi_check_img_rgb = np.array(roi_check_img_rgb, dtype=np.uint8)

                        if combine_check_img_rgb is None:
                            combine_check_img_rgb = cv2.addWeighted(check_img_rgb, 1.0,
                                                                    roi_check_img_rgb,
                                                                    0.7, 0)
                        else:
                            combine_check_img_rgb = cv2.addWeighted(combine_check_img_rgb, 1.0,
                                                                    roi_check_img_rgb,
                                                                    0.7, 0)
                        pred_roi_color = set1[mask_channel]
                        pred_roi_check_img = (mask_pred[sample_i, mask_channel, :, :] > 0.5)
                        dice = np.zeros((len(batch_pt_id)))
                        dice[sample_i] = dice_cal_fun(channel=mask_channel)(
                            batch_data[1][sample_i: sample_i + 1].cpu(),
                            pred[sample_i: sample_i + 1].cpu()).item()

                        pred_roi_check_img_rgb = np.repeat(pred_roi_check_img[:, :, np.newaxis], 3,
                                                           axis=2)
                        pred_roi_check_img_rgb = pred_roi_check_img_rgb * np.array(pred_roi_color,
                                                                                   dtype=int)
                        pred_roi_check_img_rgb = np.array(pred_roi_check_img_rgb, dtype=np.uint8)

                        if combine_pred_img_rgb is None:
                            combine_pred_img_rgb = cv2.addWeighted(check_img_rgb, 1.0,
                                                                   pred_roi_check_img_rgb,
                                                                   0.7, 0)
                        else:
                            combine_pred_img_rgb = cv2.addWeighted(combine_pred_img_rgb, 1.0,
                                                                   pred_roi_check_img_rgb,
                                                                   0.7, 0)

                    combine_out = np.zeros(
                        (2 * combine_check_img_rgb.shape[0], combine_check_img_rgb.shape[1], 3))
                    combine_out[0: combine_check_img_rgb.shape[0], :, :] = combine_check_img_rgb
                    combine_out[combine_check_img_rgb.shape[0]: 2 * combine_check_img_rgb.shape[0], :,
                    :] = combine_pred_img_rgb
                    dice_i = str(dice[sample_i])
                    file_name = '[{0:s}]_b[{1:s}]_s[{2:s}]_dice[{3:s}]img.png'.format(
                        pt_id, str(batch_i), str(sample_i), dice_i)
                    file_path = check_img_path + os.sep + file_name
                    cv2.imwrite(file_path, combine_out)

    def _shape_check_and_rescale(self, voxel, des_dim_y, des_dim_x):
        if np.ndim(voxel) == 3:
            vox_dim_y, vox_dim_x, vox_dim_z = voxel.shape
            if des_dim_y != vox_dim_y or des_dim_x != vox_dim_x:
                vox_data_type = voxel.dtype
                zoom_par = [des_dim_y / vox_dim_y, des_dim_x / vox_dim_x, 1]
                new_voxel = zoom(voxel, zoom_par, output=vox_data_type, order=1)
                return new_voxel
            else:
                return voxel

        if np.ndim(voxel) == 4:
            vox_dim_y, vox_dim_x, vox_dim_z, roi_dim = voxel.shape
            if des_dim_y != vox_dim_y or des_dim_x != vox_dim_x:
                vox_data_type = voxel.dtype
                zoom_par = [des_dim_y / vox_dim_y, des_dim_x / vox_dim_x, 1, 1]
                new_voxel = zoom(voxel, zoom_par, output=vox_data_type, order=1)
                return new_voxel
            else:
                return voxel

    def predict_numpy_pinnacle(self, img_img, output_folder=None, save_to_file=False):
        # this method predict one patient with one parameter

        # image transform
        slice_img = img_img / (1024.0 + 2048.0)
        if slice_img.shape[0] != 512 or slice_img.shape[1] != 512:
            slice_img = self._shape_check_and_rescale(slice_img, 512, 512)

        logger.info('finish scale')

        mask = np.zeros([slice_img.shape[2]] + [slice_img.shape[0]] + [slice_img.shape[1]] + [8], dtype=np.float)
        x_dim = slice_img.shape[0]
        y_dim = slice_img.shape[1]
        z_dim = slice_img.shape[2]
        dose = np.zeros([slice_img.shape[2]] + [slice_img.shape[0]] + [slice_img.shape[1]], dtype=float)
        slice_dim = 5
        half_dim = int((slice_dim - 1) / 2)
        for z in range(z_dim):
            logger.info('slice: ' + str(z))
            prediction_img = np.zeros(shape=(x_dim, y_dim, slice_dim), dtype=np.float)
            # avoid out the matrix
            for rel_slice_loc in range(slice_dim):
                z_loc = max(z + rel_slice_loc - half_dim, 0)
                z_loc = min(z_loc, z_dim - 1)
                prediction_img[:, :, rel_slice_loc] = slice_img[:, :, z_loc]
            prediction_img = np.expand_dims(prediction_img, axis=0)
            predcition_out = self.model.predict(prediction_img, batch_size=1)
            dose[z, :, :] = np.array(predcition_out[0][0, :, :, 0], dtype=float)
            mask[z, :, :, :] = np.array(predcition_out[1][0, :, :, :] > 0.5, dtype=int)

        if save_to_file:
            mask_file_name = output_folder + os.sep + 'mask.npy'
            np.save(mask_file_name, mask)
        return dose, mask

    def predict_pinnacle_all(self, pinnacle_folder):
        # this method predict all patients
        logger.info('start pinnacle prediction!')
        img_header, img_img = read_pinnacle_img(pinnacle_folder)
        self._model_set()

        [dose, masks] = self.predict_numpy_pinnacle(img_img, output_folder=pinnacle_folder)

        # write back to Pinnacle RTst
        if img_header['x_dim'] != 512 or img_header['y_dim'] != 512:
            mask_zoomed = np.zeros((masks.shape[0], img_header['x_dim'], img_header['y_dim'], 8))
            for mask_i, mask in enumerate(masks):
                logger.info('scale back:' + str(mask_i))
                mask_zoomed[mask_i] = cv2.resize(mask, dsize=(img_header['x_dim'], img_header['y_dim']),
                                                 interpolation=cv2.INTER_LINEAR)
            masks = mask_zoomed > 0.5

        logger.info('start pinnacle structure transform!')
        write_pinnacle_roi(pinnacle_folder,
                           img_header=img_header,
                           masks=masks,
                           roi_list=self.roi_list,
                           roi_2_roi_id=self.channel_info)

        pre_dvh = self.dvh_cal(masks, dose)
        np.save(pinnacle_folder + os.sep + 'dose', dose)
        # write pinnacle script
        write_pinnacle_script(pinnacle_folder, pre_dvh)

    def dvh_cal(self, roi_mask, dose_matrix):

        dvh = []
        # fig, ax = plt.subplots()
        for roi_index in range(1, roi_mask.shape[3]):
            dose_array_img_trans = dose_matrix
            dose_value = dose_array_img_trans[np.where(roi_mask[:, :, :, roi_index] == 1)]

            dose_bins = np.arange(0, 1, 0.01)
            dvh_diff_density, edges = np.histogram(dose_value, dose_bins, density=True)
            dvh_cum_density = np.cumsum(dvh_diff_density[::-1])[::-1]

            dvh.append(dvh_cum_density)

        return dvh










