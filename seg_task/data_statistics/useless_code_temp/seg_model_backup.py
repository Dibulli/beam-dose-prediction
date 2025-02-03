import logging
import os

import numpy as np
from torchsummary import summary
import torch
from torch import optim
from torch.utils.data import DataLoader

from lib.utilities import to_abs_path
from seg_task.data_statistics.useless_code_temp.seg_data import SegData
from lib.network.diss_loss import dice_loss_fun

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
        self.mask_loss_fun = None
        self.loss_fun = None

        # roi parameters
        self.roi_dicts = {0: 'CTV', 1: 'LF', 2: 'RF', 3: 'BLADDER', 4: 'PTV'}
        self.roi_number = len(self.roi_dicts)
        self.roi_loss_weight = [1] * self.cfg.channels_of_output_mask

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
            else:
                self.model_type = 'unet'
                from lib.network.unet import UNet
        else:
            self.model_type = 'unet'
            from lib.network.unet import UNet

        self.model = UNet(n_channels=1, add_channels=1, n_classes=5)
        summary(self.model.to(self.device), (1, 512, 512))

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
        epoch_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        total_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        total_sample_size = 0

        # loop batch
        for batch_i, batch_data in enumerate(tr_loader):
            imgs = batch_data[0]
            mask = batch_data[1]
            dose = batch_data[2]

            sample_size = imgs.size(0)
            total_sample_size = total_sample_size + sample_size

            if self.gpu:
                imgs = imgs.to(self.device)
                dose = dose.to(self.device)
                mask = mask.to(self.device)

            pred_out = self.model(imgs)
            mask_pred = pred_out[0]
            dose_pred = pred_out[1]

            roi_loss_value_batch = np.zeros((1, self.roi_number), dtype=float)
            loss_roi = None
            for roi_channel in self.roi_dicts:
                roi_loss_dice = dice_loss_fun(channel=roi_channel)(mask_pred, mask)
                # roi_loss_bce = torch.nn.BCELoss()(mask_pred[:, roi_channel, :, :], mask[:, roi_channel, :, :])
                roi_loss = roi_loss_dice
                roi_loss_value_batch[0, roi_channel] = roi_loss.item()

                if loss_roi is None:
                    loss_roi = roi_loss * self.roi_loss_weight[roi_channel]
                else:
                    loss_roi = loss_roi + roi_loss * self.roi_loss_weight[roi_channel]

            loss_dose = torch.nn.L1Loss()(dose_pred, dose)

            loss = loss_roi + 5 * loss_dose
            total_roi_loss = total_roi_loss + roi_loss_value_batch * sample_size
            epoch_roi_loss = total_roi_loss / total_sample_size
            total_loss = total_loss + loss.item() * sample_size
            epoch_loss = total_loss / total_sample_size

            logger.info('epoch {0:4d} training {1:.3f} --- loss: {2:.3f}'.format(
                self._current_epoch, total_sample_size / self.train_ds.sample_num, epoch_loss))

            roi_loss_str = ''
            for roi_channel in self.roi_dicts:
                roi_loss_str = roi_loss_str + '{0:s}:{1:.3f} '.format(
                    self.roi_dicts[roi_channel], epoch_roi_loss[0, roi_channel])
            logger.info(roi_loss_str)
            logger.info('{0:s}:{1:.3f} '.format(
                'loss_dose', loss_dose.item()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if 'use_adaptive_roi_weight' in self.cfg:
            pass

        return epoch_loss, epoch_roi_loss

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
        epoch_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        total_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        total_sample_size = 0

        # loop batch
        with torch.no_grad():
            for batch_i, batch_data in enumerate(val_loader):
                imgs = batch_data[0]
                mask = batch_data[1]
                dose = batch_data[2]

                sample_size = imgs.size(0)
                total_sample_size = total_sample_size + sample_size

                if self.gpu:
                    imgs = imgs.to(self.device)
                    dose = dose.to(self.device)
                    mask = mask.to(self.device)

                pred_out = self.model(imgs)
                mask_pred = pred_out[0]
                dose_pred = pred_out[1]

                roi_loss_value_batch = np.zeros((1, self.roi_number), dtype=float)
                loss_roi = None
                for roi_channel in self.roi_dicts:
                    roi_loss = dice_loss_fun(channel=roi_channel)(mask_pred, mask)
                    roi_loss_value_batch[0, roi_channel] = roi_loss.item()

                    if loss_roi is None:
                        loss_roi = roi_loss * self.roi_loss_weight[roi_channel]
                    else:
                        loss_roi = loss_roi + roi_loss * self.roi_loss_weight[roi_channel]

                loss_dose = self.loss_fun(dose_pred, dose)

                loss = loss_roi + 5 * loss_dose
                total_roi_loss = total_roi_loss + roi_loss_value_batch * sample_size
                epoch_roi_loss = total_roi_loss / total_sample_size
                total_loss = total_loss + loss.item() * sample_size
                epoch_loss = total_loss / total_sample_size

                logger.info('epoch {0:4d} training {1:.3f} --- loss: {2:.3f}'.format(
                    self._current_epoch, total_sample_size / self.train_ds.sample_num, epoch_loss))

                roi_loss_str = ''
                for roi_channel in self.roi_dicts:
                    roi_loss_str = roi_loss_str + '{0:s}:{1:.3f} '.format(
                        self.roi_dicts[roi_channel], epoch_roi_loss[0, roi_channel])
                logger.info(roi_loss_str)
                logger.info('{0:s}:{1:.3f} '.format(
                        'loss_dose', loss_dose.item()))

        return epoch_loss, epoch_roi_loss

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
                for roi_ind_in_conf, single_roi_conf in enumerate(self.roi_dicts):
                    if 'weight' in single_roi_conf:
                        self.roi_loss_weight[single_roi_conf.mask_channel] = single_roi_conf.weight
        else:
            self.loss_fun = dice_loss_fun
            loss_str = 'dice'

        self.loss_fun = torch.nn.L1Loss()

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
                csv_file.write('epoch,learning rate,epoch_loss,val_loss,')
                for roi_channel in self.roi_dicts:
                    csv_file.write(self.roi_dicts[roi_channel] + ',')
                for roi_channel in self.roi_dicts:
                    csv_file.write('val_' + self.roi_dicts[roi_channel] + ',')
                csv_file.write('\n')
        else:
            with open(os.getcwd() + os.sep + "train_loss.csv", 'a+') as csv_file:
                csv_file.write('epoch,learning rate,epoch_loss,')
                for roi_channel in self.roi_dicts:
                    csv_file.write(self.roi_dicts[roi_channel] + ',')
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
            tr_loss = self.train_run()

            # change learning rate
            scheduler.step()

            if do_validation:

                # validation

                val_loss = self.val_run()

                # write out training loss
                with open(os.getcwd() + os.sep + "train_loss.csv", 'a+') as csv_file:

                    csv_file.write('{},{},{},{},'.format(epoch, scheduler.get_lr()[0], tr_loss, val_loss))
                    csv_file.write('\n')

                tr_loss_out = tr_loss[0]
                val_loss_out = val_loss[0]
                logger.info(
                    f'Epoch: {self._current_epoch:4d} |'
                    f' Train Loss: {tr_loss_out:.4f} |'
                    f'Validation Loss: {val_loss_out:.4f}')
                roi_loss_str = 'Train ROI loss:'
                for roi_channel in self.roi_dicts:
                    roi_loss_str = roi_loss_str + '{0:s}:{1:.3f} '.format(
                        self.roi_dicts[roi_channel], tr_loss[1][0][roi_channel])
                logger.info(roi_loss_str)
                roi_loss_str = 'Validation ROI loss:'
                for roi_channel in self.roi_dicts:
                    roi_loss_str = roi_loss_str + '{0:s}:{1:.3f} '.format(
                        self.roi_dicts[roi_channel], val_loss[1][0][roi_channel])
                logger.info(roi_loss_str)
            else:
                # write out training loss without validation
                with open(os.getcwd() + os.sep + "train_loss.csv", 'a+') as csv_file:

                    csv_file.write('{},{},{},'.format(epoch, scheduler.get_lr()[0], tr_loss))
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
