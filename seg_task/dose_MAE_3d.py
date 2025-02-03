import numpy as np
import h5py as h5
import einops
from sklearn.metrics import mean_absolute_error as MAE
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt
import hydra
import yaml
from seg_task.Predict_Dose_Report import DosePrint
from seg_task.data_statistics.DVH_for_predicted_dose import max_dose_calculator
from seg_task.data_statistics.DVH_for_predicted_dose import min_dose_calculator
from seg_task.data_statistics.DVH_for_predicted_dose import d5_calculator
from seg_task.data_statistics.DVH_for_predicted_dose import d95_calculator
from seg_task.data_statistics.DVH_for_predicted_dose import max_dose_calculator
from scipy.ndimage import zoom
from scipy.spatial.distance import dice

config_file = './conf_fd/npc_server.yaml'

pred_dose = np.load("/Volumes/NPC预测数据盘/单野预测2024/pred_dose/mask_pred_single/2100943743_-1.npy")
# pred_dose = np.load("/Volumes/NPC预测数据盘/AutoPlan_2024/AutoPlan_Predicted_dose/unet/2100943743_-1.npy")
cut_pred = pred_dose[:, :, 43:129, :]
# cut_pred = pred_dose[0, 0, :, :, 43:129]

# [256, 256, z, 10]
real_file = h5.File("/Volumes/NPC预测数据盘/临时存储/test_h5_256/e9278201.h5", 'r')
real_dose = real_file["slice_dose"]
masks = real_file["slice_mask"]
ptv70 = masks[:, :, :, 1]
ptv60 = masks[:, :, :, 2]
ptv54 = masks[:, :, :, 3]
brain = masks[:, :, :, 4]
cord = masks[:, :, :, 5]
left_parotid = masks[:, :, :, 6]
right_parotid = masks[:, :, :, 7]
chiasm = masks[:, :, :, 8]
optic_nerve_l = masks[:, :, :, 9]
optic_nerve_r = masks[:, :, :, 10]
larynx = masks[:, :, :, -1]
body_1d = masks[:, :, :, 0]
sample_points = np.nonzero(body_1d)
dose_shift_list = [4, 5, 6, 7, 8, 0, 1, 2, 3, -1]
shifted_dose = np.array([real_dose[:, :, :, i] for i in dose_shift_list])
slice_dose = einops.rearrange(shifted_dose, 'b h w z -> h w z b')
total_real = slice_dose[:, :, :, -1]
total_pred = cut_pred[:, :, :, -1]
# total_pred = zoom(cut_pred, (0.5, 0.5, 1), order=1)


# for i in range(10):
#     real_dose_slice = slice_dose[:, :, :, i]
#     fake_dose_slice = cut_pred[:, :, :, i]
#     true_points = real_dose_slice[sample_points]
#     pred_points = fake_dose_slice[sample_points]
#     loss = MAE(true_points, pred_points)
#     print("mae for beam [{:}] = {:.4f}".format(i, loss))

# 输入带有患者ct，rtst，true dose 的h5， 同时输入predicted dose numpy，输出DVH图。

class DVH_Generator():
    def __init__(self, cfg):
        self.roi = cfg.roi_dicts
        self.color = cfg.colorset14
        self.output = cfg.fig_output
        # self.clip_min = cfg.clip_min
        # self.clip_max = cfg.clip_max

    def h5_dose_and_mask_combine_plot(self, real_dose, pred_dose, masks):

        roi_dict = self.roi
        roi_list = []
        x_label = list(np.arange(0, 100, 1))

        if np.ndim(masks) != 4:
            print("This mask data do not have 4 dimensions!")
            return ValueError
        # for mask in mask_npy.shape[3]:
        plt.rcParams['figure.figsize'] = (12, 8)

        for channel in range(len(roi_dict)):
            roi = roi_dict[channel].dvh_name
            print("roi [{:}] is in this channel [{:}]".format(roi, channel))
            roi_list.append(roi)
            y_label_t = []
            y_label_f = []
            y_label_t.append(100)
            y_label_f.append(100)

            mask = masks[:, :, :, channel]

            mask_with_npy = mask * real_dose
            pred_masked = mask * pred_dose
            print("masked dose generate success.")

            if mask_with_npy.any() == False:
                continue
            # 判断mask是否为空。（此病人可能没有这个轮廓）

            points_cal_t = np.count_nonzero(mask_with_npy)
            points_cal_f = np.count_nonzero(pred_masked)

            max_dose_t = int(np.ceil(np.max(mask_with_npy)))
            max_dose_f = int(np.ceil(np.max(pred_masked)))

            max_dose_bio = np.max((max_dose_t, max_dose_f))
            max_dose_bio = np.min((max_dose_bio, 100))

            for i in range(1, max_dose_bio):
                count_i = np.count_nonzero(mask_with_npy > i)
                count_i_f = np.count_nonzero(pred_masked > i)

                count_percentage = count_i / points_cal_t * 100
                count_percentage_f = count_i_f / points_cal_f * 100

                save_percentage = np.round(count_percentage, 2)
                save_percentage_f = np.round(count_percentage_f, 2)

                y_label_t.append(save_percentage)
                y_label_f.append(save_percentage_f)

                print("dose [{:}] generated!".format(i))

            print("[{}] dose 1st period completed!".format(roi))

            for j in range(max_dose_bio, 100):
                y_label_t.append(0)
                y_label_f.append(0)

            print("[{}] dose 2nd period completed!".format(roi))
            x = np.array(x_label)
            y_t = np.array(y_label_t)
            y_f = np.array(y_label_f)

            plt.plot(x, y_t, c=self.color[channel], linestyle='-', marker='^', linewidth=2, markersize=3, label=roi + "-true")
            plt.plot(x, y_f, c=self.color[channel], linestyle='--', linewidth=2, label=roi + "-predict")

        plt.axis([0, 102, 0, 102])

        plt.xlabel("dose(Gy)")
        plt.ylabel("volume(%)")
        plt.legend(loc="best")
        plt.title("DVH")
        # plt.show()
        dvh_save_path = self.output
        if not os.path.exists(dvh_save_path):
            os.mkdir(dvh_save_path)
        plt.savefig(dvh_save_path  + os.sep +  "mask_pred_single.jpg", dpi = 1000)
        plt.close()
        # pd_data.to_csv(self.output + os.sep + 'dvh_values.csv')

class Dose_Rating():
    def __init__(self, input_real, input_pred, body_points):
        # 输入dose为3维dose[256, 256, z]
        self.real_dose = input_real
        self.prediction = input_pred
        self.valid_points = body_points

    def MAE_rating(self):
        real_calculator = self.real_dose[self.valid_points]
        fake_calculator = self.prediction[self.valid_points]
        MAE_loss = MAE(real_calculator, fake_calculator)
        print("mae score = {:.4f}".format(MAE_loss))
        return  MAE_loss

    def grad_calculator(self, dose):
        grad_x = np.gradient(dose)[1]
        grad_y = np.gradient(dose)[0]
        # grad_total = abs(grad_x) + abs(grad_y)

        return grad_x, grad_y
    # def DVH_rating(self, real_dose, prediction, real_masks):
        # ptv70_d5 = real_dose *
        # return  MAE_loss
    def Gradient_rating(self, eps=1e-5):
        grad_real_x, grad_real_y = self.grad_calculator(self.real_dose)
        grad_pred_x, grad_pred_y = self.grad_calculator(self.prediction)
        # print("grad MAE = {:.4f}".format(MAE(grad_real[self.valid_points], grad_pred[self.valid_points])))

        # grad_mat = np.divide(2 * grad_real * grad_pred, np.square(grad_real) + np.square(grad_pred))
        ouclid_mat = abs(grad_real_x - grad_pred_x) + abs(grad_real_y - grad_pred_y)
        ouclid_score = np.mean(ouclid_mat[self.valid_points])
        print("grad score = {:.4f}".format(ouclid_score))

        return ouclid_score

    def Dice_rating(self, thresh):
        true_total_index = self.real_dose > thresh
        true_array = einops.rearrange(true_total_index, "h w z -> (h w z)")

        pred_index = self.prediction > thresh
        pred_array = einops.rearrange(pred_index, "h w z -> (h w z)")

        sum_dice = 1 - dice(true_array, pred_array)
        print("dice score for thresh [{:}] Gy = {:.4f}".format(thresh, sum_dice))

        return sum_dice




@hydra.main(config_path=config_file)
def main(cfg):
    dg = DVH_Generator(cfg)
    dg.h5_dose_and_mask_combine_plot(real_dose = slice_dose[:, :, :, -1], pred_dose=cut_pred[:, :, :, -1], masks=masks)
    # dose_print = DosePrint(cfg)
    # for beam in range(10):
    #     dose_print.dose_glance_generate(real_dose = slice_dose, pred_dose=cut_pred, beam_index=beam)

    # dose_print.dose_glance_generate(real_dose = slice_dose, pred_dose=cut_pred, beam_index=0)

    # dr = Dose_Rating(total_real, total_pred, sample_points)
    # dr.MAE_rating()

    # for i in range(0, 10):
    #     dr = Dose_Rating(slice_dose[:, :, :, i], cut_pred[:, :, :, i], sample_points)
    #     dr.MAE_rating()

    # dr.Gradient_rating()
    # k = 0
    # for i in range(1, 11):
    #     thr = i * 7
    #     k += dr.Dice_rating(thr)
    # print("average dice score [{:.3f}]".format(k/10))
if __name__ == "__main__":
    main()