import numpy as np
import matplotlib.pyplot as plt
import pydicom
import h5py as h5
import cv2
import yaml
import logging
import os
import time
import pandas as pd

logger = logging.getLogger(__name__)

# 最后需将此模块加入到dicom_process中！！！！！！！

class BeamContourGeneration():
    def __init__(self, input_4d_numpy, patient_id=None):
        self.min_y = 0
        self.max_y = 0
        self.min_x = 0
        self.max_x = 0

        self.beam_indexes = 9
        # self.rois = 13
        self.rotate_angel = 40
        self.half_dim = 128
        self.source_point_euclid = [self.half_dim, (-884 + self.dim_x) / 2]
        self.pt_id = patient_id
        """ source point源轴距 1000 mm，转换为体素， 1000 / 1.132812 = 882.75 （Pixel Spacing = 1.132812） """

        self.input_4d_numpy = input_4d_numpy
        # self.output_5d_numpy = np.zeros((self.dim_x, self.dim_y, self.dim_z, self.beam_indexes, self.rois), dtype=bool)
        self.output_4d_beams = np.zeros((self.dim_x, self.dim_x, self.dim_z, self.beam_indexes), dtype=np.float32)
        self.output_ptv70_projection = np.zeros((self.dim_x, self.dim_x, self.dim_z, self.beam_indexes), dtype=np.float32)
        self.output_ptv60_projection = np.zeros((self.dim_x, self.dim_x, self.dim_z, self.beam_indexes), dtype=np.float32)
        self.output_ptv54_projection = np.zeros((self.dim_x, self.dim_x, self.dim_z, self.beam_indexes), dtype=np.float32)
        self.output_oars_projection = np.zeros((self.dim_x, self.dim_x, self.dim_z, self.beam_indexes), dtype=np.float32)

        self.body = self.input_4d_numpy[:, :, :, 0]
        self.ptv70 = self.input_4d_numpy[:, :, :, 1]
        self.ptv60 = self.input_4d_numpy[:, :, :, 2]
        self.ptv54 = self.input_4d_numpy[:, :, :, 3]
        self.oars = np.sum(self.input_4d_numpy[:, :, :, 4:], axis=-1, keepdims=False)

    def BeamContour2d(self, mask_npy, centx, centy, body_slice, ptv70_slice, ptv60_slice, ptv54_slice, oar_slice):
        """mask_npy 预期输入一组contour的结合，但也可能为空，开头"""
        mask_npy = np.array(mask_npy).astype("float")
        body_slice = np.array(body_slice).astype("float")
        ptv70_slice = np.array(ptv70_slice).astype("float")
        ptv60_slice = np.array(ptv60_slice).astype("float")
        ptv54_slice = np.array(ptv54_slice).astype("float")
        oar_slice = np.array(oar_slice).astype("float")

        output_3d = np.zeros((mask_npy.shape[0], mask_npy.shape[1], self.beam_indexes))
        out_ptv70_projection = np.zeros_like(output_3d)
        out_ptv60_projection = np.zeros_like(output_3d)
        out_ptv54_projection = np.zeros_like(output_3d)
        out_oar_projection = np.zeros_like(output_3d)

        slice_d1 = np.zeros_like(mask_npy)
        if mask_npy.any() == True:
            for i in range(self.beam_indexes):
                # print(i)
            # i = 1
                xi, yi = np.meshgrid(np.arange(self.dim_x), np.arange(self.dim_x))
                M = cv2.getRotationMatrix2D((centx, centy), self.rotate_angel * i, 1)
                M_inverse = cv2.getRotationMatrix2D((centx, centy), -self.rotate_angel * i, 1)
                slice_d0 = cv2.warpAffine(mask_npy, M, (self.dim_x, self.dim_x), flags=cv2.INTER_NEAREST)
                body_rotate = cv2.warpAffine(body_slice, M, (self.dim_x, self.dim_x), flags=cv2.INTER_NEAREST)
                normalize = np.max(np.sum(body_rotate, axis=0, keepdims=False))
                ptv70_rotate = cv2.warpAffine(ptv70_slice, M, (self.dim_x, self.dim_x), flags=cv2.INTER_NEAREST)
                ptv60_rotate = cv2.warpAffine(ptv60_slice, M, (self.dim_x, self.dim_x), flags=cv2.INTER_NEAREST)
                ptv54_rotate = cv2.warpAffine(ptv54_slice, M, (self.dim_x, self.dim_x), flags=cv2.INTER_NEAREST)
                oar_rotate = cv2.warpAffine(oar_slice, M, (self.dim_x, self.dim_x), flags=cv2.INTER_NEAREST)
                proj_ptv70 = np.sum(ptv70_rotate, axis=0, keepdims=True) * body_rotate / normalize
                proj_ptv60 = np.sum(ptv60_rotate, axis=0, keepdims=True) * body_rotate / normalize
                proj_ptv54 = np.sum(ptv54_rotate, axis=0, keepdims=True) * body_rotate / normalize
                proj_oar =  - np.sum(oar_rotate, axis=0, keepdims=True) * body_rotate / normalize

                # slice_d1 = np.zeros_like(mask_npy)

                # slice_d0[np.nonzero(slice_d0 < 1)] = 0
                # ⬆️这里可能有bug
                # cv2 getRotationMatrix 会默认将旋转之后的图像插值！！！

                trans_matx = slice_d0 * xi
                trans_maty = slice_d0 * yi
                if trans_matx.any() == False or trans_maty.any() == False:
                    logger.info("[{:}]this patient's data has some error!".format(self.pt_id))
                    return output_3d
                min_x = int(trans_matx[np.where(trans_matx > 0)].min() - 3)
                max_x = int(np.max(trans_matx) + 3)
                min_x_line = trans_maty[:, int(min_x + 3)]
                max_x_line = trans_maty[:, int(max_x - 3)]
                min_y = min_x_line[np.where(min_x_line > 0)].min()
                max_y = max_x_line[np.where(max_x_line > 0)].min()
                # min_y = trans_maty[np.where(trans_maty > 0)].min()
                # max_y = np.max(trans_maty)


                ind1 = np.where((yi - min_y) * (centx - min_x) - \
                                (-883 + centy  - min_y) * (xi - min_x) > 0)
                slice_d1[ind1] = 1
                ind2 = np.where((yi - max_y) * (centx - max_x) - \
                                (-883 + centy - max_y) * (xi - max_x) > 0)
                slice_d1[ind2] = 0
                # p3 = time.time()
                # print("expense [{:.3f}] seconds to value line ".format(p3 - p1))
                beam_with_pdd = self.add_pdd_to_beam(body_rotate, slice_d1)
                # slice_output = cv2.warpAffine(slice_d1, M_inverse, (256, 256))
                slice_output = cv2.warpAffine(beam_with_pdd, M_inverse, (self.dim_x, self.dim_x))
                ptv70_output = cv2.warpAffine(proj_ptv70, M_inverse, (self.dim_x, self.dim_x))
                ptv60_output = cv2.warpAffine(proj_ptv60, M_inverse, (self.dim_x, self.dim_x))
                ptv54_output = cv2.warpAffine(proj_ptv54, M_inverse, (self.dim_x, self.dim_x))
                oar_output = cv2.warpAffine(proj_oar, M_inverse, (self.dim_x, self.dim_x))

                output_3d[:, :, i] = slice_output
                out_ptv70_projection[:, :, i] = ptv70_output
                out_ptv60_projection[:, :, i] = ptv60_output
                out_ptv54_projection[:, :, i] = ptv54_output
                out_oar_projection[:, :, i] = oar_output

            return output_3d, out_ptv70_projection, out_ptv60_projection, out_ptv54_projection, out_oar_projection
        else:
            return output_3d, out_ptv70_projection, out_ptv60_projection, out_ptv54_projection, out_oar_projection

    def rotation_center(self):

        ptv60 = self.input_4d_numpy[:, :, :, 2]
        x_all, y_all = np.nonzero(ptv60)[0], np.nonzero(ptv60)[1]
        cent_x, cent_y = int(x_all.mean().round()), int(y_all.mean().round())

        return cent_x, cent_y

    def beam_contour_composition_or(self, beams_npy):
        beams_all_in_one = np.logical_or.reduce(beams_npy, 2)
        # beams_all_in_one:[1024, 1024] 的一个布尔矩阵
        return beams_all_in_one

    def add_pdd_to_beam(self, body_slice, beam_slice):
        csv_file = pd.read_csv("/data/szh/PDD.csv")
        # csv_file = pd.read_csv("/Users/mr.chai/Desktop/PDD.csv")

        # depth = csv_file.iloc[:, 0]
        dose_percentage = np.array(csv_file.iloc[:, 1]) / 100
        output_slice = np.zeros_like(body_slice, dtype=np.float32)
        x_range = np.nonzero(body_slice)[1]
        x_min = np.min(x_range)
        x_max = np.max(x_range)
        # print("xmin=[{:}], xmax=[{:}]".format(x_min, x_max))
        for x in range(x_min, x_max):
        # x=206
            piece_y = body_slice[:, x]
            # print(x)
            # print(piece_y.any())
            if piece_y.any() == False:
                return output_slice
            #     continue
            output_piece = np.zeros_like(piece_y, dtype=np.float32)
            min_y = np.nonzero(piece_y)[0].min()
            max_y = np.nonzero(piece_y)[0].max()
            distance_y = int(max_y - min_y) + 1
            output_piece[min_y:max_y] = dose_percentage[1:distance_y]
            output_slice[:, x] = output_piece

        pdd_with_beam = output_slice * beam_slice
        return pdd_with_beam

    def ptv_mask_composition(self):
        if self.input_4d_numpy.ndim == 4:
            ptv70 = self.input_4d_numpy[:, :, :, 1]
            # ptv66 = self.input_4d_numpy[:, :, :, 1]
            ptv60 = self.input_4d_numpy[:, :, :, 2]
            ptv54 = self.input_4d_numpy[:, :, :, 3]
            ptv1 = np.logical_or(ptv70, ptv60)
            # ptv2 = np.logical_or(ptv1, ptv54)
            ptv_all = np.logical_or(ptv1, ptv54)
            x_all, y_all = np.nonzero(ptv60)[0], np.nonzero(ptv60)[1]
            cent_x, cent_y = int(x_all.mean().round()), int(y_all.mean().round())
            # 输出为三维矩阵[1024, 1024, z_dim]
            return ptv_all, cent_x, cent_y
        else:
            logger.info("this input mask numpy is not 4D!")
            return ValueError

    def beam_sum(self, beams_npy):
        beams_all_in_one = np.sum(beams_npy, 2)
        return beams_all_in_one

    def output(self):
        # centx, centy = self.rotation_center()
        ptv_contours, centx, centy = self.ptv_mask_composition()

        # ptv_contours, centx, centy = self.oar_mask_composition()

        for z in range(self.dim_z):
        # z = 77
        # print(z)
            body_slice = self.body[:, :, z]
            ptv70_slice = self.ptv70[:, :, z]
            ptv60_slice = self.ptv60[:, :, z]
            ptv54_slice = self.ptv54[:, :, z]
            oar_slice = self.oars[:, :, z]
            beams, ptv70_projection, ptv60_projection, ptv54_projection, oars_projection = \
                self.BeamContour2d(ptv_contours[:, :, z], centx, centy, body_slice,
                                                ptv70_slice, ptv60_slice, ptv54_slice, oar_slice)

            self.output_4d_beams[:, :, z, :] = beams
            self.output_ptv70_projection[:, :, z, :] = ptv70_projection
            self.output_ptv60_projection[:, :, z, :] = ptv60_projection
            self.output_ptv54_projection[:, :, z, :] = ptv54_projection
            self.output_oars_projection[:, :, z, :] = oars_projection

        print("Beam contours of slice[{:}] has been created!".format(z))

        output_5d_beams = [self.output_4d_beams, self.output_ptv70_projection,
                            self.output_ptv60_projection, self.output_ptv54_projection,
                            self.output_oars_projection]
        # [256, 256, z, 9, 5]
        # return self.output_3d_numpy
        # return output_4d_beams
        return output_5d_beams
    """output_5d_numpy=[512, 512, z, 9, 13], shape(4)=0 代表0度下所有roi的beam contour。"""


def main():
    h5path = "/Volumes/NPC预测数据盘/test_data/47fabf67.h5"
    output_path = "/Volumes/NPC预测数据盘/"
    # 02209225 dose 是反的。

    h5data = h5.File(h5path, 'r')
    print(h5data.keys())
    npy = h5data["slice_mask"]
    bc = BeamContourGeneration(npy)
    out = bc.output()
    np.save(output_path + os.sep + "5d_beam_contour" + ".npy", out)
    # out = bc.oar_mask_composition()[0]


if __name__ == "__main__":
    main()

# h5path = "/Users/mr.chai/Desktop/7cac3a17.h5"
# with h5.File(h5path, "w") as hf:
#     hf.create_dataset("beam_contour", data=dst, compression='lzf')

