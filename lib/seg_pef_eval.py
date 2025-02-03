import logging
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

logger = logging.getLogger(__name__)


def calc_performance(mask_a, mask_b, pixel_spacing=(0.8, 0.8), slice_thickness=5, mode='volume'):
    """
hd, hd95, ahd, dice, jaccard
    :param mask_a: input 1 [height, width, slice_number]
    :param mask_b: input 2 [height, width, slice_number]
    :param mode: 'volume' or 'slice'
    :param pixel_spacing: pixel spacing
    :param slice_thickness: thickness
    :return: HD, ASD, DICE, JACCARD
            [1] for 'volume' mode, [1, slice_number] for 'slice' mode
    """

    # the shape of mask should be 4D
    assert mask_a.ndim == 3
    assert mask_b.ndim == 3
    assert mask_a.shape == mask_b.shape

    # two model, "volume" or "slice"
    assert mode == 'slice' or mode == 'volume'
    if mode == 'volume':
        assert isinstance(slice_thickness, (int, float))

    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    slice_number = mask_a.shape[2]

    if mode == 'volume':
        # deal with all zero condition
        if ~np.any(mask_a) and ~np.any(mask_b):
            return 0, 0, 0, 1, 1
        if ~np.any(mask_a) or ~np.any(mask_b):
            return 999, 999, 999, 0, 0

        mask_a_dist = distance_transform_edt(np.logical_not(mask_a),
                                             sampling=(pixel_spacing[0], pixel_spacing[1], slice_thickness))
        mask_b_dist = distance_transform_edt(np.logical_not(mask_b),
                                             sampling=(pixel_spacing[0], pixel_spacing[1], slice_thickness))

        hd_dist = np.concatenate((mask_a_dist[mask_b], mask_b_dist[mask_a]))
        hd = np.amax(hd_dist)
        hd95 = np.percentile(hd_dist, 95)
        ahd = np.mean(hd_dist)

        # dice and Jaccard
        intersection = np.sum(mask_a * mask_b)
        dice = (2. * intersection) / (np.sum(mask_a) + np.sum(mask_b))
        jaccard = intersection / (np.sum(mask_a) + np.sum(mask_b) - intersection)

        return hd, hd95, ahd, dice, jaccard

    if mode == 'slice':
        hd = []
        hd95 = []
        ahd = []
        dice = []
        jaccard = []

        # Hausdorff distance, average surface distance
        for slice_i in range(slice_number):

            # deal with all zero condition
            if ~np.any(mask_a[:, :, slice_i]) and ~np.any(mask_b[:, :, slice_i]):
                hd.append(0)
                ahd.append(0)
                dice.append(1)
                jaccard.append(1)
            elif ~np.any(mask_a[:, :, slice_i]) or ~np.any(mask_b[:, :, slice_i]):
                hd.append(None)
                ahd.append(None)
                dice.append(0)
                jaccard.append(0)
            else:
                mask_a_dist = distance_transform_edt(np.logical_not(mask_a[:, :, slice_i]),
                                                     sampling=(pixel_spacing[0], pixel_spacing[1]))
                mask_b_dist = distance_transform_edt(np.logical_not(mask_b[:, :, slice_i]),
                                                     sampling=(pixel_spacing[0], pixel_spacing[1]))

                hd_dist = np.concatenate((mask_a_dist[mask_b], mask_b_dist[mask_a]))
                hd95.append(np.percentile(hd_dist, 95))
                hd.append(np.amax(hd_dist))
                ahd.append(np.mean(hd_dist))

                # dice and Jaccard
                intersection = np.sum(mask_a[:, :, slice_i] * mask_b[:, :, slice_i])
                dice.append((2. * intersection) / (np.sum(mask_a[:, :, slice_i]) + np.sum(mask_b[:, :, slice_i])))
                jaccard.append(intersection /
                               (np.sum(mask_a[:, :, slice_i]) + np.sum(mask_b[:, :, slice_i]) - intersection))

        return