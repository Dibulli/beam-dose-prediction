import logging
import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary

from lib.utilities import to_abs_path
from seg_data import SegData
from lib.network.diss_loss import dice_loss_fun
from lib_version1.network import segformer_pytorch as segformer

logger = logging.getLogger(__name__)


class SegModel:
    def __init__(self, cfg):
        self.cfg = cfg

        # dataset
        self.train_ds = None
        self.val_ds = None

        # model
        self.model_type = 'unet'
        self.model = None
        self.model_file_path = to_abs_path(cfg.model_file_path)

        # for training
        self._current_epoch = 0
        self.optimizer = None
        self.loss_fun = None

        # roi parameters
        self.roi_dicts = cfg.roi_dicts
        self.roi_number = len(self.roi_dicts)
        # self.roi_loss_weight = [1] * self.cfg.channels_of_middle_images


        if cfg.cpu_or_gpu == 'gpu':
            self.gpu = True
            self.device = torch.device('cuda:' + str(cfg.gpu_id))
        else:
            self.gpu = False
            self.device = 'cpu'

    def model_build(self):
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
            elif self.cfg.model_type == "Segformer":
                self.model_type = "Segformer"
            else:
                self.model_type = 'unet'
                from lib.network.unet import UNet
        else:
            self.model_type = 'unet'
            from lib.network.unet import UNet

        # self.model = UNet(self.cfg.channels_of_input_images, self.cfg.channels_of_middle_images, self.cfg.channels_of_output_mask)
        # self.model = UNet(self.cfg.channels_of_input_images + self.cfg.channels_of_input_masks,
        #                   self.cfg.channels_of_output_mask)
        self.model = segformer.Segformer()
#        summary(self.model.to(self.device), (6, 512, 512))

        # set device before build
        if self.gpu:
            self.model.to(self.device)

    def load_weight(self):
        logger.info('load weight')
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))
        logger.info('load weight successful')

    def train_run(self):
        # set model to train
        self.model.train()

        # set the loaders
        tr_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size,
                               num_workers=self.cfg.train_loader_works,
                               pin_memory=True)

        # initial training loss
        epoch_loss = 0
        total_loss = 0
        loss_dose = 0
        loss = 0
        # epoch_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        # total_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        total_sample_size = 0

        # loop batch
        for batch_i, batch_data in enumerate(tr_loader):
            imgs = batch_data[0]
            np_masks = batch_data[1]
            true_dose = batch_data[2]
            img_with_mask = batch_data[5]


            # print("img_shape:", imgs.shape)
            # print(img_with_mask.shape)
            # print("true_masks_shape:", true_masks.shape)
            true_dose_array = np.array(true_dose)
            # pos = np.unravel_index(np.argmax(true_dose_array), true_dose_array.shape)
            b = np.max(true_dose_array)
            # print("true_dose_max:", true_dose_array[pos])
            # print("true_dose_max:{0: .3f}".format(true_dose_array[pos]))
            logger.info("true_dose max in all slices: {0: .3f}".format(b))
            logger.info("true_middle layer: {0: .4f} -- {1: .4f} -- {2: .4f}".format(true_dose[0, 0, 254, 254],
                                                                                    true_dose[0, 0, 255, 255],
                                                                                    true_dose[0, 0, 256, 256]))

            # print("true_middle layer: {0: .4f} -- {1: .4f} -- {2: .4f}".format(true_dose[0, 0, 254, 254],
            #                                                                    true_dose[0, 0, 255, 255],
            #                                                                    true_dose[0, 0, 256, 256]))
            sample_size = imgs.size(0)
            total_sample_size = total_sample_size + sample_size

            if self.gpu:
                img_with_mask = img_with_mask.to(self.device)
                np_masks = np_masks.to(self.device)
                # true_masks = true_masks.to(self.device)
                true_dose = true_dose.to(self.device)

            pred_out = self.model(img_with_mask)
            # print("true_dose.shape:", true_dose.shape)
            # masks_pred = pred_out[0]
            dose_pred = pred_out
            a = dose_pred.cpu().detach().numpy()
            # posa = np.unravel_index(np.argmax(a), a.shape)
            # print("pre_dose_max:{0: .3f}".format(a[posa]))
            logger.info("pred_dose max in all slices: {0: .3f}".format(np.max(a)))
            logger.info("pre_middle layer: {0: .4f} -- {1: .4f} -- {2: .4f}".format(a[0, 0, 254, 254],
                                                                                    a[0, 0, 255, 255],
                                                                                    a[0, 0, 256, 256]))
            # print("pre_middle layer: {0: .4f} -- {1: .4f} -- {2: .4f}".format(a[0, 0, 254, 254],
            #                                                                   a[0, 0, 255, 255],
            #                                                                   a[0, 0, 256, 256]))
            # # print("dose_pred_shape:", a.shape)
            # print("dose_pred_type:", type(a))
            # print("dose_pred_max:", a[pos])

            """loss calculation edited by szh 4.27"""
            true_dose_3d = true_dose[:, 0, :, :]
            pred_dose_3d = pred_out[:, 0, :, :]
            true_mat = torch.zeros_like(np_masks)
            pred_mat = torch.zeros_like(np_masks)
            loss_scalar = 0

            for roi_ind_in_conf, single_roi_conf in enumerate(self.cfg.roi_dicts):

                mask_i = np_masks[:, roi_ind_in_conf, :, :]
                true_dose_with_mask = torch.where(mask_i == 1, true_dose_3d, mask_i)
                pred_dose_with_mask = torch.where(mask_i == 1, pred_dose_3d, mask_i)
                true_mat[:, roi_ind_in_conf, :, :] = true_dose_with_mask
                pred_mat[:, roi_ind_in_conf, :, :] = pred_dose_with_mask

            loss_dose = torch.nn.L1Loss()(pred_mat, true_mat)
            loss_without_mask = torch.nn.L1Loss()(pred_out, true_dose)
            loss = loss_dose + loss_without_mask
            # loss_dose = torch.nn.L1Loss()(pred_out, true_dose)
            # mask_nonezero = torch.nonzero(mask_i, as_tuple=True)

            # 怎么判断索引为空？？？？？
            # true_dose_with_mask = mask_i
            # pred_dose_with_mask = mask_i
                # if mask_nonezero[0].size() != torch.Size([0]):
                #     print("nonzero")
                #     true_dose_with_mask[mask_nonezero] = true_dose[mask_nonezero]
                #     pred_dose_with_mask[mask_nonezero] = pred_out[mask_nonezero]

                # ⬆️loss_dose 的type：tensor(0.2931, grad_fn=<L1LossBackward>)

            # roi_loss_value_batch = np.zeros((1, self.roi_number), dtype=float)
            # loss_roi = None
            # for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
            #     # print("mask_pred.shape:", masks_pred.shape)
            #     # print("true_masks.shape:", true_masks.shape)
            #     # print("dose_pred.shape:", dose_pred.shape)
            #     # print("true_dose.shape:", true_dose.shape)
            #
            #     roi_loss_dice = self.loss_fun(channel=single_roi_conf.mask_channel)(masks_pred, true_masks)
            #     roi_loss_bce = torch.nn.BCELoss()(masks_pred[:, single_roi_conf.mask_channel, :, :], true_masks[:, single_roi_conf.mask_channel, :, :])
            #     roi_loss = roi_loss_dice + roi_loss_bce
            #     roi_loss_value_batch[0, roi_ind_in_conf] = roi_loss.item()
            #
            #     if loss_roi is None:
            #         loss_roi = roi_loss * self.roi_loss_weight[single_roi_conf.mask_channel]
            #     else:
            #         loss_roi = loss_roi + roi_loss * self.roi_loss_weight[single_roi_conf.mask_channel]
            # print("dose_pred_shape:", dose_pred.shape)
            # print("true_dose_shape:", true_dose.shape)
            # loss = loss_roi + 10 * loss_dose
            #
            # total_roi_loss = total_roi_loss + roi_loss_value_batch * sample_size
            # epoch_roi_loss = total_roi_loss / total_sample_size
            total_loss = total_loss + loss.item() * sample_size
            epoch_loss = total_loss / total_sample_size


            logger.info('epoch {0:4d} training {1:.3f} --- loss: {2:.3f}'.format(
                self._current_epoch, total_sample_size / self.train_ds.sample_num, epoch_loss))
            # roi_loss_str = ''
            # for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
            #     roi_loss_str = roi_loss_str + '{0:s}:{1:.3f} '.format(
            #         single_roi_conf.name, epoch_roi_loss[0, roi_ind_in_conf])
            # logger.info(roi_loss_str)
            logger.info('{0:s}:{1:.3f} '.format(
                'loss_dose', loss.item()))
            logger.info("")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # if 'use_adaptive_roi_weight' in self.cfg:
        #     if self.cfg.use_adaptive_roi_weight:
        #         for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
        #             self.roi_loss_weight[single_roi_conf.mask_channel] = 1 / epoch_roi_loss[0, roi_ind_in_conf]

        # return epoch_loss, epoch_roi_loss, loss_dose
        return epoch_loss, loss

    def val_run(self):
        # set model to evaluation
        self.model.eval()

        # set the loaders
        val_loader = DataLoader(self.val_ds, batch_size=self.cfg.batch_size,
                                num_workers=self.cfg.validation_load_works,
                                pin_memory=True)

        # initial training loss
        epoch_loss = 0
        total_loss = 0
        # epoch_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        # total_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        total_sample_size = 0

        # loop batch
        with torch.no_grad():
            for batch_i, batch_data in enumerate(val_loader):

                imgs = batch_data[0]
                np_masks = batch_data[1]
                true_dose = batch_data[2]
                img_with_mask = batch_data[5]

                sample_size = imgs.size(0)
                total_sample_size = total_sample_size + sample_size

                if self.gpu:
                    img_with_mask = img_with_mask.to(self.device)
                    np_masks = np_masks.to(self.device)
                    true_dose = true_dose.to(self.device)

                pred_out = self.model(img_with_mask)
                # masks_pred = pred_out[0]
                dose_pred = pred_out

                """loss calculation edited by szh 4.27"""
                true_dose_3d = true_dose[:, 0, :, :]
                pred_dose_3d = pred_out[:, 0, :, :]
                true_mat = torch.zeros_like(np_masks)
                pred_mat = torch.zeros_like(np_masks)
                # loss_scalar = 0

                for roi_ind_in_conf, single_roi_conf in enumerate(self.cfg.roi_dicts):
                    mask_i = np_masks[:, roi_ind_in_conf, :, :]
                    true_dose_with_mask = torch.where(mask_i == 1, true_dose_3d, mask_i)
                    pred_dose_with_mask = torch.where(mask_i == 1, pred_dose_3d, mask_i)
                    true_mat[:, roi_ind_in_conf, :, :] = true_dose_with_mask
                    pred_mat[:, roi_ind_in_conf, :, :] = pred_dose_with_mask

                loss_dose = torch.nn.L1Loss()(pred_mat, true_mat)
                loss_without_mask = torch.nn.L1Loss()(pred_out, true_dose)
                loss = loss_dose + loss_without_mask
                # roi_loss_value_batch = np.zeros((1, self.roi_number), dtype=float)

                # loss_roi = None
                # for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
                #     roi_loss_dice = self.loss_fun(channel=single_roi_conf.mask_channel)(masks_pred, true_masks)
                #     roi_loss_bce = torch.nn.BCELoss()(masks_pred[:, single_roi_conf.mask_channel, :, :],
                #                                       true_masks[:, single_roi_conf.mask_channel, :, :])
                #     roi_loss = roi_loss_dice + roi_loss_bce
                #     roi_loss_value_batch[0, roi_ind_in_conf] = roi_loss.item()
                #
                #     if loss_roi is None:
                #         loss_roi = roi_loss * self.roi_loss_weight[single_roi_conf.mask_channel]
                #     else:
                #         loss_roi = loss_roi + roi_loss * self.roi_loss_weight[single_roi_conf.mask_channel]

                # loss_dose = torch.nn.L1Loss()(dose_pred, true_dose)
                # loss = loss_roi + 10 * loss_dose
                # print("val_loss:", loss)
                # print("val_loss_dose:", loss_dose)

                # total_roi_loss = total_roi_loss + roi_loss_value_batch * sample_size
                # epoch_roi_loss = total_roi_loss / total_sample_size
                total_loss = total_loss + loss.item() * sample_size
                epoch_loss = total_loss / total_sample_size

                # print("val_total_roi_loss:", total_roi_loss)
                # print("val_epoch_roi_loss:", epoch_roi_loss)
                # print("val_total_loss:", total_loss)
                # print("val_epoch_loss:", epoch_loss)

                logger.info('epoch {0:4d} validation {1:.3f} --- loss: {2:.3f}'.format(
                    self._current_epoch, total_sample_size / self.val_ds.sample_num, epoch_loss))
                # roi_loss_str = ''
                # for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
                #     roi_loss_str = roi_loss_str + '{0:s}:{1:.3f} '.format(
                #         single_roi_conf.name, epoch_roi_loss[0, roi_ind_in_conf])
                # logger.info(roi_loss_str)
                logger.info('{0:s}:{1:.3f} '.format(
                    'loss_dose', loss.item()))

        # return epoch_loss, epoch_roi_loss, loss_dose
        return epoch_loss, loss


    def SGD_initial(self, weight_decay):
        optimizer_str = 'SGD'
        if 'momentum' in self.cfg:
            momentum = self.cfg.momentum
        else:
            momentum = 0

        if 'nesterov' in self.cfg:
            nesterov = self.cfg.nesterov
        else:
            nesterov = False
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.cfg.learn_rate,
                                   weight_decay=weight_decay,
                                   momentum=momentum,
                                   nesterov=nesterov)
        return optimizer_str, momentum, nesterov

    def train_start(self):
        # initial training dataset
        self.train_ds = SegData(self.cfg, 'train')

        # set optimizer
        if 'weight_decay' in self.cfg:
            weight_decay = self.cfg.weight_decay
        else:
            weight_decay = 0

        if 'optimizer' in self.cfg:
            if self.cfg.optimizer == 'RMSprop':
                optimizer_str = 'RMSprop'
                self.optimizer = optim.RMSprop(self.model.parameters(),
                                               lr=self.cfg.learn_rate,
                                               weight_decay=weight_decay)
            elif self.cfg.optimizer == 'Adam':
                optimizer_str = 'Adam'
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=self.cfg.learn_rate,
                                            weight_decay=weight_decay)
            else:
                optimizer_str, momentum, nesterov = self.SGD_initial(weight_decay)
        else:
            optimizer_str, momentum, nesterov = self.SGD_initial(weight_decay)

        # set loss
        if 'loss' in self.cfg:
            if self.cfg.loss == 'tversky':
                from lib.network.tversky_loss import tversky_loss_fun_factory
                loss_tversky_alpha = float(self.cfg.tversky_alpha)
                loss_tversky_beta = float(self.cfg.tversky_beta)
                self.loss_fun = tversky_loss_fun_factory(alpha=loss_tversky_alpha, beta=loss_tversky_beta)
                loss_str = 'tversky'
            elif self.cfg.loss == 'topk_bce':
                from lib.network.topk_bce_loss import topk_bce_loss_fun_factory
                loss_str = 'topk_bce'
            elif self.cfg.loss == 'topk_bce_x_dice':
                from lib.network.topk_bce_x_dice_loss import topk_bce__x_dice_loss_fun_factory
                loss_topk_bce_k = float(self.cfg.topk_bce_k)
                self.loss_fun = topk_bce__x_dice_loss_fun_factory(k=loss_topk_bce_k)
                loss_str = 'topk_bce_x_dice'
            elif self.cfg.loss == 'focal_loss':
                from lib.network.focal_loss import focal_loss_factory
                self.loss_fun = focal_loss_factory
                loss_str = 'focal_loss'
            else:
                self.loss_fun = dice_loss_fun
                loss_str = 'dice'
                # read roi weight from cfg
                # for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
                #     if 'weight' in single_roi_conf:
                #         self.roi_loss_weight[single_roi_conf.mask_channel] = single_roi_conf.weight
        else:
            self.loss_fun = dice_loss_fun
            loss_str = 'dice'

        # read roi weight from cfg
        # for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
        #     if 'weight' in single_roi_conf:
        #         self.roi_loss_weight[single_roi_conf.mask_channel] = single_roi_conf.weight

        logger.info('this phase epoch number   : ' + str(self.cfg.epochs))
        logger.info('train image series number : ' + str(self.train_ds.patient_num))
        logger.info('train sample number       : ' + str(self.train_ds.sample_num))
        logger.info('batch size                : ' + str(self.cfg.batch_size))
        logger.info('network                   : ' + self.model_type)
        logger.info('initial learning rate     : ' + str(self.cfg.learn_rate))
        logger.info('decay epoch               : ' + str(self.cfg.lr_decay_epoch))
        logger.info('decay rate                : ' + str(self.cfg.lr_decay_rate))
        logger.info('save period               : ' + str(self.cfg.save_period))

        logger.info('optimizer                 : ' + optimizer_str)
        if optimizer_str == 'SGD':
            logger.info('momentum                  : ' + str(momentum))
            logger.info('nesterov                  : ' + str(nesterov))
        logger.info('weight decay              : ' + str(weight_decay))

        logger.info('loss                      : ' + loss_str)
        if loss_str == 'tversky':
            logger.info('loss alpha                : ' + str(loss_tversky_alpha))
            logger.info('loss beta                 : ' + str(loss_tversky_beta))

        # set learning rate decay
        self.optimizer.step()
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.cfg.lr_decay_epoch, gamma=self.cfg.lr_decay_rate)

        do_validation = self.cfg.do_validation
        if do_validation:
            # read validation dataset
            self.val_ds = SegData(self.cfg, 'validation')
            if self.val_ds.sample_num == 0:
                logger.warning('not have validation data')
                do_validation = False

        # check validation setting
        if do_validation:
            logger.info('have validation?          : ' + str(do_validation))
            logger.info('validation patient number : ' + str(self.val_ds.patient_num))
            logger.info('validation sample number  : ' + str(self.val_ds.sample_num))
        else:
            logger.info('have validation?          : ' + str(do_validation))
            logger.warning('only training!')

        # head line of the csv file
        if do_validation:
            with open(os.getcwd() + os.sep + "train_loss.csv", 'a+') as csv_file:
                # csv_file.write('epoch,learning rate,epoch_loss,epoch_val_loss,dose_loss,dose_val_loss')
                csv_file.write('epoch, learning rate, dose_loss, dose_val_loss')

                # for single_roi_conf in self.roi_dicts:
                #     csv_file.write(single_roi_conf.name + ',')
                # for single_roi_conf in self.roi_dicts:
                #     csv_file.write('val_' + single_roi_conf.name + ',')
                csv_file.write('\n')
        else:
            with open(os.getcwd() + os.sep + "train_loss.csv", 'a+') as csv_file:
                csv_file.write('epoch,learning rate,epoch_loss,dose_loss')
                # for single_roi_conf in self.roi_dicts:
                #     csv_file.write(single_roi_conf.name + ',')
                csv_file.write('\n')

        logger.info('TRAINING START!!!')

        # loop epoch
        for epoch in range(self.cfg.epochs):
            logger.debug('Starting epoch {}/{}.'.format(epoch + 1, self.cfg.epochs))
            self._current_epoch = epoch

            if 'loss' in self.cfg:
                if self.cfg.loss == 'topk_bce':
                    # k decay
                    loss_topk_bce_k = float(self.cfg.topk_bce_k)
                    current_k = 0.98 ** epoch * loss_topk_bce_k
                    logger.info('loss k                    : ' + str(current_k))
                    self.loss_fun = topk_bce_loss_fun_factory(k=current_k)

            # train
            tr_loss, tr_dose_loss = self.train_run()

            # change learning rate
            # scheduler.step()

            if do_validation:

                # validation
                with torch.no_grad():
                    val_loss, val_dose_loss = self.val_run()

                # write out training loss
                with open(os.getcwd() + os.sep + "train_loss.csv", 'a+') as csv_file:

                    # csv_file.write('{},{},{},{},{},{},'.format(epoch, scheduler.get_lr()[0], tr_loss, val_loss,
                    #                                           tr_dose_loss, val_dose_loss))
                    csv_file.write('{},{},{},{}'.format(epoch, scheduler.get_lr()[0], tr_dose_loss, val_dose_loss))
                    # for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
                    #     csv_file.write(str(tr_roi_loss[0, roi_ind_in_conf]) + ',')
                    # for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
                    #     csv_file.write(str(val_roi_loss[0, roi_ind_in_conf]) + ',')
                    csv_file.write('\n')

                logger.info(
                    f'Epoch: {self._current_epoch:4d} |'
                    f' Train Loss: {tr_loss:.4f} |'
                    f'Validation Loss: {val_loss:.4f}')
            else:
                # write out training loss without validation
                with open(os.getcwd() + os.sep + "train_loss.csv", 'a+') as csv_file:

                    # csv_file.write('{},{},{},{}'.format(epoch, scheduler.get_lr()[0], tr_loss, tr_dose_loss))
                    csv_file.write('{},{},{}'.format(epoch, scheduler.get_lr()[0], tr_dose_loss))

                    # for internal_roi_i, roi in enumerate(self.roi_dicts):
                    #     csv_file.write(str(tr_roi_loss[0, internal_roi_i]) + ',')
                    csv_file.write('\n')

                logger.info('Epoch: {0:4d} | Train Loss: {1:.4f}'.format(
                    self._current_epoch, tr_loss))

            # save model
            if epoch % self.cfg.save_period == 0:
                model_path = os.path.dirname(self.model_file_path)
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                torch.save(self.model.state_dict(), self.model_file_path)
                logger.debug('Checkpoint {} saved !'.format(epoch))

    def train_all(self):
        self.model_build()
        if self.cfg.continue_train:
            self.load_weight()
        self.train_start()
