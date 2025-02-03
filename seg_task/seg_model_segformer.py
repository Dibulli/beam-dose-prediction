import logging
import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary

from lib.utilities import to_abs_path
# from segformer.seg_data_segformer import SegData
from seg_data_segformer import SegData

from lib.network.diss_loss import dice_loss_fun
from lib.network import segformer_pytorch as segformer
# from lib.network.unet import UNet
from torch.autograd import Variable

from torch.nn.functional import kl_div
import gc
from torchvision.transforms.functional import rotate
import einops


logger = logging.getLogger(__name__)


class SegModel:
    def __init__(self, cfg):
        self.cfg = cfg

        # dataset
        self.train_ds = None
        self.val_ds = None

        # model
        self.model_type = 'unet'
        self.modelA = None

        # model with Discriminator
        # self.model_D = None


        # A=200, B=240, C=280, .... H=120, I=160
        self.model_file_path = to_abs_path(cfg.model_file_path)

        self.cos_array = np.array([1,  0.76604444,  0.17364818, -0.5, -0.93969262, -0.93969262, -0.5,  0.17364818,  0.76604444])
        self.sin_array = np.array([0,  0.64278761,  0.98480775,  0.8660254,  0.34202014,
                                   -0.34202014, -0.8660254 , -0.98480775, -0.64278761])
        # for training
        self._current_epoch = 0
        self.optimizerA = None

        # self.optimizer_D = None

        # Calculate output of image discriminator (PatchGAN)
        self.patch = (1, cfg.dim_x // 2 ** 4, cfg.dim_y // 2 ** 4)
        """patch:[9, 256/16, 256/16]"""

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
            elif self.cfg.model_type == "segformer":
                self.model_type = "segformer"
            else:
                self.model_type = 'unet'
                from lib.network.unet import UNet
        else:
            self.model_type = 'unet'
            from lib.network.unet import UNet


        self.modelA = UNet(n_channels=6, n_classes=1)

        # self.model_D = segformer.Segformer(channels=9)

        # set device before build
        if self.gpu:
            self.modelA.to(self.device)

            # self.model_D.to(self.device)


    """梯度矩阵归一， 除以100"""
    def array_normalization(self, input_array):
        min_grad = torch.min(input_array)
        # max_grad = torch.max(input_array)
        # c = max_grad - min_grad
        return  input_array / 10 - min_grad / 10

    """所有masks逆时针旋转9个角度，生成的9个tensor"""
    def masks_rotation_9beams(self, masks, cent_x, cent_y):
        bc = masks.size(0)
        mask_channel = masks.size(1)
        dim_x = masks.size(2)
        dim_y = masks.size(3)
        trans_masks = torch.zeros((9, bc, mask_channel, dim_x, dim_y))
        for beam_index in range(9):
            angle = 20 + 40 * beam_index
            for mask in range(mask_channel):
                mask_slice = masks[:, mask, :, :]
                mask_rotation = rotate(mask_slice, angle, center=[cent_x, cent_y])
                trans_masks[beam_index, :, mask, :, :] = mask_rotation
        return trans_masks
    #     返回[9， batch size， mask channel， 512， 512]，[0, :, :, :, :]代表200度下的beam。

    """9个分野dose按不同角度统一旋转至0 位置"""
    def tensor_rotatation_in_9_beams(self, true_dose_tensor, centx, centy):
        """true dose tensor:[batch size, 9, 512, 512]"""
        trans_tensor = torch.zeros_like(true_dose_tensor)
        beam_index = true_dose_tensor.size(1)
        for i in range(beam_index):
            angle = 20 + 40 * i
            dose_slice = true_dose_tensor[:, i, :, :]
            rotation = rotate(dose_slice, angle, center=[centx, centy])
            trans_tensor[:, i, :, :] = rotation
        return trans_tensor

    """9个分野dose从0位置逆向旋转至该分野角度"""
    def tensor_inverse_rotatation(self, true_dose_tensor, centx, centy):
        """true dose tensor:[batch size, 9, 512, 512]"""
        trans_tensor = torch.zeros_like(true_dose_tensor)
        beam_index = true_dose_tensor.size(1)
        for i in range(beam_index):
            angle = - 20 - 40 * i
            dose_slice = true_dose_tensor[:, i, :, :]
            rotation = rotate(dose_slice, angle, center=[centx, centy])
            trans_tensor[:, i, :, :] = rotation
        return trans_tensor

    """img和contour按照9个beam依次旋转至0位置"""
    def gan_input_generation(self, imgs, contours, cent_x, cent_y):
        trans_tensor = torch.zeros((imgs.size(0), 2, contours.size(1), imgs.size(2), imgs.size(3)))
        """trans tensor : [1, 2, 9, 512, 512], contour index 5 相当于true dose 的 index 0。"""

        beam_index = contours.size(1)
        for i in range(beam_index):
            angel = 20 + 40 * i
            contour = contours[:, i, :, :].unsqueeze(1)
            rotation_img = rotate(imgs, angel, center=[cent_x, cent_y])
            rotation_contour = rotate(contour, angel, center=[cent_x, cent_y])
            gan_input_i = torch.cat((rotation_img, rotation_contour), dim=1)
            # [1, 2, 512, 512]
            trans_tensor[:, :, i, :, :] = gan_input_i

        return trans_tensor


    def load_weight(self):
        logger.info('load weight')
        check_point = torch.load(self.model_file_path, map_location=self.device)
        self.modelA.load_state_dict(check_point['modelA'])
        # self.model_D.load_state_dict(check_point['model_D'])
        logger.info('load weight successful')

    def train_run(self):
        # set model to train
        self.modelA.train()

        # self.model_D.train()

        # set the loaders
        tr_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size,
                               num_workers=self.cfg.train_loader_works,
                               pin_memory=True)

        # initial training loss

        loss = 0
        # cos_array = torch.from_numpy(self.cos_array)
        # sin_array =  torch.from_numpy(self.sin_array)
        total_sample_size = 0

        # loop batch
        for batch_i, batch_data in enumerate(tr_loader):
            imgs = batch_data[0]
            beams = batch_data[1]
            # masks = batch_data[2]
            true_dose = batch_data[3]
            # true_dose = batch_data[2]

            """
            imgs = [batch size, 1, 512, 512]
            masks = [batch size, 14, 512, 512]
            beams = [batch szie, 9, 512, 512]
            doses = [batch size, 10, 512, 512]

            """
            """criteria for GAN loss calculation"""
            # valid = Variable(torch.FloatTensor(np.ones((9, *self.patch))), requires_grad=False)
            # fake = Variable(torch.FloatTensor(np.zeros((9, *self.patch))), requires_grad=False)
            # noise = Variable(torch.randn((9, 1, 256, 256)), requires_grad = False)

            if self.gpu:
                imgs = imgs.to(self.device)
                true_dose = true_dose.to(self.device)
                beams = beams.to(self.device)
                # valid = valid.to(self.device)
                # fake = fake.to(self.device)
                # masks = masks.to(self.device)
                # cos_array = cos_array.to(self.device)
                # sin_array = sin_array.to(self.device)
                # noise = noise.to(self.device)
            single_true_dose = true_dose[:, :-1, :, :]
            # [1, 9, 512, 512]
            total_true_dose = true_dose[:, -1:, :, :]
            # total_true_dose = einops.rearrange(true_dose[:, --1, :, :], '(b c) h w -> b c h w', c=1)
            """contours [:, :, :, :, 5], shape = [b, 13, h, w, 9] 表示的是200度的射野，dose[: 0, :, :]表示的是200度的剂量"""
            """body=[1, 1, 512, 512]"""
            # true_9_dose_diverse = true_dose[:, :-1, :, :]
            # rotated_inputs = torch.zeros(
            #     (imgs.size(0), 2 + masks.size(1), contours.size(1), imgs.size(2), imgs.size(3))).to(self.device)
            # rotated_dose = torch.zeros_like(true_9_dose_diverse).unsqueeze(1).to(self.device)
            # for i in range(contours.size(1)):
            #     img_with_contour = torch.cat((imgs, contours[:, i, :, :].unsqueeze(1)), dim=1)
            #     img_contour_mask = torch.cat((img_with_contour, masks), dim=1)
            #     rotated_inputs[:, :, i, :, :] = rotate(img_contour_mask, 40 * i)
            #     rotated_dose[:, :, i, :, :] = rotate(true_9_dose_diverse[:, i, :, :].unsqueeze(1), 40 * (i + 5))
            """rotated inputs - 0: 0度下的beam；rotated dose - 0: 原200度的dose 旋转到0度"""

            """img with contour: [1, 2, 9, 512, 512]"""
            sample_size = imgs.size(0)
            total_sample_size = total_sample_size + sample_size
            imgs = einops.repeat(imgs, 'b c h w -> (repeat b) c h w', repeat=9)
            # [9, 1, 512, 512]

            # masks = einops.repeat(masks, 'b c h w -> (repeat b) c h w', repeat=9)
            # [9, 14, 512, 512]
            # body = masks[:, :1, :, :]
            # ptvs = torch.sum(masks[:, 1:4, :, :], dim=1, keepdim=True)
            # oars = torch.sum(masks[:, 4:, :, :], dim=1, keepdim=True)
            # reshape_masks = torch.cat((body, ptvs, oars), dim=1)
            # beams = einops.rearrange(beams, 'c b h w -> b c h w')
            beams = einops.rearrange(beams, 'i b c h w -> (b i) c h w')

            # [9, 5, 512, 512]

            doses = einops.rearrange(single_true_dose, 'c b h w -> b c h w')
            # [9, 1, 512, 512]
            # std=1
            # mean=0
            # noise = torch.randn(imgs.size()) * std + mean
            # noise.to(self.device)
            # body = einops.rearrange(masks[:, 0, :, :], "(b c) h w -> b c h w", c=1)
            # x_input = torch.cat((imgs, beams), dim=1)
            # transformer_input_ib = torch.cat((imgs, beams, masks), dim=1).float()
            # transformer_input_ib = torch.cat((imgs, beams, reshape_masks), dim=1).float()
            transformer_input_ib = torch.cat((imgs, beams), dim=1).float()

            # [9, 6, 512, 512]
            """input=[1, 14, 512, 512]"""

            pred_single = self.modelA(transformer_input_ib)
            # input_single = einops.rearrange(pred_single, 'b c h w -> c b h w')
            # pred_total = self.model_D(input_single)
            # loss = torch.nn.L1Loss()(pred_total, total_true_dose)
            loss = torch.nn.L1Loss()(pred_single, doses)
            # kl_loss = kl_div(pred_single.softmax(dim=0).log(), doses.softmax(dim=0))
            # pred_all = torch.sum(pred_single, dim=0, keepdim=True)

            # total_loss = torch.nn.L1Loss()(pred_all, total_true_dose)
            # loss = 10 * single_dose_loss +  total_loss
            # ptv_pred = pred * ptvs
            # ptv_real = doses * ptvs
            # oar_pred = pred * oars
            # oar_real = doses * oars

            # ptv_loss = torch.nn.L1Loss()(ptv_pred, ptv_real)
            # oar_loss = torch.nn.L1Loss()(oar_pred, oar_real)
            # true_dose_grad_x = torch.gradient(doses, dim=(2, 3))[1]
            # true_dose_grad_y = torch.gradient(doses, dim=(2, 3))[0]

            # pred_dose_grad_x = torch.gradient(pred, dim=(2, 3))[1]
            # pred_dose_grad_y = torch.gradient(pred, dim=(2, 3))[0]

            # reshape_true_grad_x = einops.rearrange(true_dose_grad_x, 'b c h w->c h w b')
            # reshape_true_grad_y = einops.rearrange(true_dose_grad_y, 'b c h w->c h w b')
            # reshape_pred_grad_x = einops.rearrange(pred_dose_grad_x, 'b c h w->c h w b')
            # reshape_pred_grad_y = einops.rearrange(pred_dose_grad_y, 'b c h w->c h w b')

            # true_grad_axis = reshape_true_grad_x * cos_array + reshape_true_grad_y * sin_array
            # true_grad_trans = - reshape_true_grad_x * sin_array + reshape_true_grad_y * cos_array
            # pred_grad_axis = reshape_pred_grad_x * cos_array + reshape_pred_grad_y * sin_array
            # pred_grad_trans = - reshape_pred_grad_x * sin_array + reshape_pred_grad_y * cos_array

            # true_grad_axis_orig_shape = einops.rearrange(true_grad_axis.detach(), 'c h w b -> b c h w')
            # true_grad_trans_orig_shape = einops.rearrange(true_grad_trans.detach(), 'c h w b -> b c h w')
            # pred_grad_axis_orig_shape = einops.rearrange(pred_grad_axis.detach(), 'c h w b -> b c h w')
            # pred_grad_trans_orig_shape = einops.rearrange(pred_grad_trans.detach(), 'c h w b -> b c h w')
            # [9, 1, 512, 512]

            # aug_grad_t = (true_grad_trans_orig_shape + noise) * body
            # aug_grad_a = (true_grad_axis_orig_shape + noise) * body

            # gan_input_true = torch.cat((imgs, true_grad_axis_orig_shape, true_grad_trans_orig_shape), dim=1).float()
            # gan_input_fake = torch.cat((imgs, pred_grad_axis_orig_shape, pred_grad_trans_orig_shape), dim=1).float()
            # _real = torch.cat((imgs, aug_grad_a, aug_grad_t), dim=1).float()
            # gan_input_true = torch.cat((true_grad_axis_orig_shape, true_grad_trans_orig_shape), dim=1).float()
            # gan_input_fake = torch.cat((pred_grad_axis_orig_shape, pred_grad_trans_orig_shape), dim=1).float()

            # [9, 3, 512, 512]
            # 如果计算loss错误，可能要把梯度的维度变换回去。c h w b -> b c h w
            # pred_real = self.model_D(gan_input_true)
            # pred_fake = self.model_D(gan_input_fake)
            # _pred_real = self.model_D(_real)
            # loss_G = torch.nn.L1Loss()(pred_fake, valid)
            # _loss_d = torch.nn.L1Loss()(_pred_real, fake)
            # mask_loss = torch.nn.L1Loss()(mask_pred, mask_true)
            # total_dose_loss = torch.nn.L1Loss()(pred_all, total_true_dose)
            # loss =  10 * single_dose_loss + 5 * total_dose_loss +  loss_G
            # loss =  10 * single_dose_loss +  10 * total_dose_loss

            # loss = 0.1 * single_dose_loss + total_dose_loss +  loss_G + _loss_d


            self.optimizerA.zero_grad()
            loss.backward()
            self.optimizerA.step()

            # loss_real = torch.nn.L1Loss()(pred_real, valid)
            # loss_fake = torch.nn.L1Loss()(pred_fake.detach(), fake)
            # loss_D = 0.5 * (loss_real + loss_fake)
            # self.optimizer_D.zero_grad()
            # loss_D.backward()
            # self.optimizer_D.step()

            logger.info('epoch {0:4d} training {1:.3f}'.format(
                self._current_epoch, total_sample_size / self.train_ds.sample_num))

            logger.info(
                # 'loss G = {:.3f} / '
                '/loss = {:.3f}'
                # 'kl_loss = {:.3f}'
                # '/single loss = {:.3f}'
                # '/total loss = {:.3f}'
                # '/loss_fake = {:.3f}'
                    .format(loss.item()))
                    # loss_D.item(),
                    # _loss_d.item(),
                    # single_dose_loss.item(),
                    # total_dose_loss.item()))
        # return loss_G.item(), loss_D.item(), loss.item(), _loss_d.item(), loss_fake.item()
        return loss.item()

    def val_run(self):
        # set model to evaluation
        self.modelA.eval()

        # self.model_D.eval()

        # set the loaders
        val_loader = DataLoader(self.val_ds, batch_size=self.cfg.batch_size,
                               num_workers=self.cfg.validation_load_works,
                               pin_memory=True)

        # initial training loss
        # cos_array = torch.from_numpy(self.cos_array)
        # sin_array = torch.from_numpy(self.sin_array)
        # epoch_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        # total_roi_loss = np.zeros((1, self.roi_number), dtype=float)
        total_sample_size = 0

        # loop batch
        for batch_i, batch_data in enumerate(val_loader):
            imgs = batch_data[0]
            beams = batch_data[1]
            # masks = batch_data[2]
            true_dose = batch_data[3]
            # true_dose = batch_data[2]

            """
            imgs = [batch size, 1, 512, 512]
            masks = [batch size, 14, 512, 512]
            beams = [batch szie, 9, 512, 512]
            doses = [batch size, 10, 512, 512]

            """
            """criteria for GAN loss calculation"""
            # valid = Variable(torch.FloatTensor(np.ones((9, *self.patch))), requires_grad=False)
            # fake = Variable(torch.FloatTensor(np.zeros((9, *self.patch))), requires_grad=False)
            # noise = Variable(torch.randn((9, 1, 256, 256)), requires_grad = False)
            if self.gpu:
                imgs = imgs.to(self.device)
                true_dose = true_dose.to(self.device)
                beams = beams.to(self.device)
                # valid = valid.to(self.device)
                # fake = fake.to(self.device)
                # masks = masks.to(self.device)
                # cos_array = cos_array.to(self.device)
                # sin_array = sin_array.to(self.device)
                # noise = noise.to(self.device)
            single_true_dose = true_dose[:, :-1, :, :]
            # [1, 9, 512, 512]

            # total_true_dose = einops.rearrange(true_dose[:, -1, :, :], '(b c) h w -> b c h w', c=1)
            total_true_dose = true_dose[:, -1:, :, :]

            """contours [:, :, :, :, 5], shape = [b, 13, h, w, 9] 表示的是200度的射野，dose[: 0, :, :]表示的是200度的剂量"""
            """body=[1, 1, 512, 512]"""
            # true_9_dose_diverse = true_dose[:, :-1, :, :]
            # rotated_inputs = torch.zeros(
            #     (imgs.size(0), 2 + masks.size(1), contours.size(1), imgs.size(2), imgs.size(3))).to(self.device)
            # rotated_dose = torch.zeros_like(true_9_dose_diverse).unsqueeze(1).to(self.device)
            # for i in range(contours.size(1)):
            #     img_with_contour = torch.cat((imgs, contours[:, i, :, :].unsqueeze(1)), dim=1)
            #     img_contour_mask = torch.cat((img_with_contour, masks), dim=1)
            #     rotated_inputs[:, :, i, :, :] = rotate(img_contour_mask, 40 * i)
            #     rotated_dose[:, :, i, :, :] = rotate(true_9_dose_diverse[:, i, :, :].unsqueeze(1), 40 * (i + 5))
            """rotated inputs - 0: 0度下的beam；rotated dose - 0: 原200度的dose 旋转到0度"""

            """img with contour: [1, 2, 9, 512, 512]"""
            sample_size = imgs.size(0)
            total_sample_size = total_sample_size + sample_size
            imgs = einops.repeat(imgs, 'b c h w -> (repeat b) c h w', repeat=9)
            # [9, 1, 512, 512]

            # masks = einops.repeat(masks, 'b c h w -> (repeat b) c h w', repeat=9)
            # [9, 14, 512, 512]
            # body = masks[:, :1, :, :]
            # ptvs = masks[:, 1:4, :, :]
            # oars = torch.sum(masks[:, 4:, :, :], dim=1, keepdim=True)

            # reshape_masks = torch.cat((body, ptvs, oars), dim=1)

            # beams = einops.rearrange(beams, 'c b h w -> b c h w')
            beams = einops.rearrange(beams, 'i b c h w -> (b i) c h w')

            # [9, 1, 512, 512]

            doses = einops.rearrange(single_true_dose, 'c b h w -> b c h w')
            # [9, 1, 512, 512]
            # std=1
            # mean=0
            # noise = torch.randn(imgs.size()) * std + mean
            # noise.to(self.device)
            # body = einops.rearrange(masks[:, 0, :, :], "(b c) h w -> b c h w", c=1)
            # x_input = torch.cat((imgs, beams), dim=1)
            # transformer_input_ib = torch.cat((imgs, beams, masks), dim=1).float()
            # transformer_input_ib = torch.cat((imgs, beams, reshape_masks), dim=1).float()
            transformer_input_ib = torch.cat((imgs, beams), dim=1).float()

            # [9, 2, 512, 512]
            """input=[1, 14, 512, 512]"""

            pred_single = self.modelA(transformer_input_ib)
            loss = torch.nn.L1Loss()(pred_single, doses)

            # kl_loss = kl_div(pred_single.softmax(dim=0).log(), doses.softmax(dim=0))
            # pred_all = torch.sum(pred_single, dim=0, keepdim=True)

            # total_loss = torch.nn.L1Loss()(pred_all, total_true_dose)
            # loss = 10 * single_dose_loss + total_loss
            # ptv_pred = pred * ptvs
            # ptv_real = doses * ptvs
            # oar_pred = pred * oars
            # oar_real = doses * oars

            # ptv_loss = torch.nn.L1Loss()(ptv_pred, ptv_real)
            # oar_loss = torch.nn.L1Loss()(oar_pred, oar_real)
            # true_dose_grad_x = torch.gradient(doses, dim=(2, 3))[1]
            # true_dose_grad_y = torch.gradient(doses, dim=(2, 3))[0]

            # pred_dose_grad_x = torch.gradient(pred, dim=(2, 3))[1]
            # pred_dose_grad_y = torch.gradient(pred, dim=(2, 3))[0]

            # reshape_true_grad_x = einops.rearrange(true_dose_grad_x, 'b c h w->c h w b')
            # reshape_true_grad_y = einops.rearrange(true_dose_grad_y, 'b c h w->c h w b')
            # reshape_pred_grad_x = einops.rearrange(pred_dose_grad_x, 'b c h w->c h w b')
            # reshape_pred_grad_y = einops.rearrange(pred_dose_grad_y, 'b c h w->c h w b')

            # true_grad_axis = reshape_true_grad_x * cos_array + reshape_true_grad_y * sin_array
            # true_grad_trans = - reshape_true_grad_x * sin_array + reshape_true_grad_y * cos_array
            # pred_grad_axis = reshape_pred_grad_x * cos_array + reshape_pred_grad_y * sin_array
            # pred_grad_trans = - reshape_pred_grad_x * sin_array + reshape_pred_grad_y * cos_array

            # loss_grad_axis = torch.nn.L1Loss()(pred_grad_axis, true_grad_axis)
            # loss_grad_trans = torch.nn.L1Loss()(pred_grad_trans, true_grad_trans)
            # loss_grad = loss_grad_axis + loss_grad_trans

            # true_grad_axis_orig_shape = einops.rearrange(true_grad_axis.detach(), 'c h w b -> b c h w')
            # true_grad_trans_orig_shape = einops.rearrange(true_grad_trans.detach(), 'c h w b -> b c h w')
            # pred_grad_axis_orig_shape = einops.rearrange(pred_grad_axis.detach(), 'c h w b -> b c h w')
            # pred_grad_trans_orig_shape = einops.rearrange(pred_grad_trans.detach(), 'c h w b -> b c h w')
            # [9, 1, 512, 512]

            # aug_grad_t = (true_grad_txis_orig_shape + noise) * body
            # aug_grad_a = (true_grad_arans_orig_shape + noise) * body

            # gan_input_true = torch.cat((imgs, true_grad_axis_orig_shape, true_grad_trans_orig_shape), dim=1).float()
            # gan_input_fake = torch.cat((imgs, pred_grad_axis_orig_shape, pred_grad_trans_orig_shape), dim=1).float()
            # _real = torch.cat((imgs, aug_grad_a, aug_grad_t), dim=1).float()
            # gan_input_true = torch.cat((true_grad_axis_orig_shape, true_grad_trans_orig_shape), dim=1).float()
            # gan_input_fake = torch.cat((pred_grad_axis_orig_shape, pred_grad_trans_orig_shape), dim=1).float()
            # [9, 3, 512, 512]
            # 如果计算loss错误，可能要把梯度的维度变换回去。c h w b -> b c h w
            # pred_real = self.model_D(gan_input_true)
            # pred_fake = self.model_D(gan_input_fake)
            # _pred_real = self.model_D(_real)
            # _loss_d = torch.nn.L1Loss()(_pred_real, fake)
            # loss_G = torch.nn.L1Loss()(pred_fake, valid)
            # single_dose_loss = torch.nn.L1Loss()(pred, doses)

            # pred_all =  torch.sum(pred, dim=0, keepdim=True)
            # mask_pred = pred_all * masks[:, :3, :, :]
            # mask_true = total_true_dose * masks[:, :3, :, :]
            # total_dose_loss = torch.nn.L1Loss()(pred_all, total_true_dose)
            # mask_loss = torch.nn.L1Loss()(mask_pred, mask_true)
            # loss = 0.1 * single_dose_loss + total_dose_loss +  loss_G + _loss_d
            # loss = 10 * single_dose_loss + 5 * total_dose_loss +  loss_G
            # loss = 10 * single_dose_loss + 10 * total_dose_loss


            # loss_real = torch.nn.L1Loss()(pred_real, valid)
            # loss_fake = torch.nn.L1Loss()(pred_fake.detach(), fake)
            # loss_D = 0.5 * (loss_real + loss_fake)

            logger.info('epoch {0:4d} validation {1:.3f}'.format(
                self._current_epoch, total_sample_size / self.val_ds.sample_num))

            logger.info(
                # 'loss G = {:.3f} / '
                # 'kl_loss = {:.3f}'
                '/ loss = {:.3f}'
                # '/single loss = {:.3f}'
                # '/total loss = {:.3f}'
                    .format(
                    # loss_G.item(),
                    loss.item()))
                    # single_dose_loss.item(),
                    # _loss_d.item(),
                    # total_dose_loss.item()))
            # return loss_G.item(), loss_D.item(), loss.item(), loss_real.item(), total_dose_loss.item()
        return loss.item()
            # return loss_G.item(), loss_D.item(), loss.item(), _loss_d.item(), loss_fake.item()

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
        self.optimizerA = optim.SGD(self.modelA.parameters(),
                                   lr=self.cfg.learn_rate,
                                   weight_decay=weight_decay,
                                   momentum=momentum,
                                   nesterov=nesterov)

        """gan optimizer"""
        # self.optimizer_D = optim.SGD(self.model_D.parameters(),
        #                            lr=self.cfg.learn_rate,
        #                            weight_decay=weight_decay,
        #                            momentum=momentum,
        #                            nesterov=nesterov)

        return optimizer_str, momentum, nesterov
    def add_gaussian_noise(self, tensor, body_3d, mean=0, std=1):
        body = einops.rearrange(body_3d, "(b c) h w -> b c h w", c=1)
        noise = torch.randn(tensor.size()) * std + mean
        noise.to(self.device)
        noisy_tensor = tensor + noise
        valid_tensor = noisy_tensor * body
        return valid_tensor

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
                self.optimizerA = optim.RMSprop(self.modelA.parameters(),
                                               lr=self.cfg.learn_rate,
                                               weight_decay=weight_decay)


                # self.optimizer_D = optim.RMSprop(self.model_D.parameters(),
                #                                lr=self.cfg.learn_rate,
                #                                weight_decay=weight_decay)

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
        self.optimizerA.step()
        # self.optimizer_D.step()


        schedulerA = optim.lr_scheduler.StepLR(self.optimizerA, self.cfg.lr_decay_epoch, gamma=self.cfg.lr_decay_rate)
        # scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, self.cfg.lr_decay_epoch, gamma=self.cfg.lr_decay_rate)

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
                csv_file.write('epoch, learning rate, '
                               'loss, val_loss')
                               # ', single loss, val_single_loss, total loss, val_total_loss,')
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
            tr_loss = self.train_run()

            # change learning rate
            schedulerA.step()
            # scheduler_D.step()

            if do_validation:
                # validation
                with torch.no_grad():
                    val_loss = self.val_run()

                # write out training loss
                with open(os.getcwd() + os.sep + "train_loss.csv", 'a+') as csv_file:

                    # csv_file.write('{},{},{},{},{},{},'.format(epoch, scheduler.get_lr()[0], tr_loss, val_loss,
                    #                                           tr_dose_loss, val_dose_loss))
                    csv_file.write('{},{},{},{}'.format(epoch, schedulerA.get_lr()[0],
                                                                    tr_loss, val_loss))
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
                    csv_file.write('{},{},{}'.format(epoch, schedulerA.get_lr()[0], tr_loss_G))

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
                torch.save({'modelA': self.modelA.state_dict(),
                            # 'model_D': self.model_D.state_dict(),
                            }, self.model_file_path)
                logger.debug('Checkpoint {} saved !'.format(epoch))

    def train_all(self):
        self.model_build()
        if self.cfg.continue_train:
            self.load_weight()
        self.train_start()
