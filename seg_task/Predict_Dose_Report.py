from lib.dicom.dicom_dose_series import DicomDoseSeries
from lib.dicom.dicom_image_series import DicomCTSeries
from lib.dicom.dicom_directory import DicomDirectory
import numpy as np
import matplotlib.pyplot as plt
import os
import hydra
from scipy.ndimage import zoom

config_file = 'conf_fd/npc_server.yaml'


class DosePrint():
    def __init__(self, cfg):
        self.box_size = cfg.box_size
        self.out = cfg.dose_glance_output
        if not os.path.exists(self.out):
            os.mkdir(self.out)

    def dose_padding(self, input_dose_npy):
        x_dim = input_dose_npy.shape[0]
        y_dim = input_dose_npy.shape[1]
        z_dim = input_dose_npy.shape[2]

        cut_dose = input_dose_npy[1:x_dim - 1, 1:y_dim - 1, :]
        pad_dose = np.pad(cut_dose, 1, 'constant', constant_values=100)
        output_dose_npy = pad_dose[:, :, 1:z_dim + 1]

        z_cmpl = (z_dim // self.box_size + 1) * self.box_size
        padd_z = np.zeros([x_dim, y_dim, z_cmpl])
        padd_z[:, :, 0:z_dim] = output_dose_npy

        return padd_z

    def dose_glance_generate(self, real_dose, pred_dose, beam_index):
        box = self.box_size
        # for beam_index in range(10):
        # fig_vmax = 80
        if beam_index == 9:
            fig_vmax = 80
        else:
            fig_vmax = 20
        true = real_dose[64:192, 64:192, :, beam_index]
        pred = pred_dose[64:192, 64:192, :, beam_index]
        dif = np.abs(true - pred)

        true_vis = self.dose_padding(true)
        pred_vis = self.dose_padding(pred)
        dif_vis = self.dose_padding(dif)
        x_dim = true_vis.shape[0]
        y_dim = true_vis.shape[1]
        z_dim = true_vis.shape[2]
        # padding 之后的维度，希望在这里把z_dim取整
        up_mat = np.zeros([x_dim * 3, y_dim * box])
        down_mat = np.zeros_like(up_mat)

        z_group = z_dim // box
        # 6
        z_clip = z_group * box
        # 60

        for z in range(z_group):
            if z % 2 == 0:
                for ind in range(box * z, box * z + box):
                    TruePredMinus = np.concatenate([true_vis[:, :, ind], pred_vis[:, :, ind], dif_vis[:, :, ind]], axis=0)
                    up_mat[0: 3*x_dim, (ind%box)*y_dim:(ind%box)* y_dim + y_dim] = TruePredMinus
                continue
            else:
                for ind in range(box * z, box * z + box):
                    TruePredMinus = np.concatenate([true_vis[:, :, ind], pred_vis[:, :, ind], dif_vis[:, :, ind]], axis=0)
                    down_mat[0: 3 * x_dim, (ind%box) * y_dim:(ind%box)* y_dim + y_dim] = TruePredMinus
            out_mat = np.concatenate([up_mat, down_mat], axis=0)
            plt.imshow(out_mat, cmap='jet', vmin=0, vmax=fig_vmax)
            plt.show()
            # plt.savefig(self.out + os.sep + "beam[" + str(beam_index) + "]_slice[" + str(z) + "].jpg", dpi = 1000, bbox_inches = 'tight')
            plt.close()
        print("All glance images were saved!")

#     如果predict的目录不存在可能会报错。

@hydra.main(config_path=config_file)
def main(cfg, real_dose, pred_dose, beam):
    dp = DosePrint(cfg)
    dp.dose_glance_generate(real_dose, pred_dose, beam)

if __name__ == "__main__":
    main()
