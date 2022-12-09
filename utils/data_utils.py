"""data_utils.py: Everything for data augmentation on VerSe dataset."""

__author__ = "Luis Serrador"

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # uncomment to not use GPU

import numpy as np
import math

from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import distance_transform_edt as distance


def calc_centr(A):
    """
    Calcute the center of coordinates A

    :param A: coordinates - NumPy array of shape [3, n]
    :return: center of the coordinates - NumPy array
    """

    length = len(A[0])
    sum_slice = np.sum(A[0])
    sum_h = np.sum(A[1])
    sum_w = np.sum(A[2])

    if sum_slice == 0:
        return np.array([0, 0, 0]).astype(int)
    else:
        return np.array([np.around(sum_slice / length), np.around(sum_h / length), np.around(sum_w / length)]).astype(
            int)


def calc_centr_vertebras(mask_after_resamp, vert):
    """
    Calculate center of vertebra

    :param mask_after_resamp: mask - NumPy array
    :param vert: vert id - float or int
    :return: center of vertebra
    """
    return calc_centr(np.where(mask_after_resamp == vert))


def calc_dist_map(seg):
    """
    Calcute the distance map of all pixels to the border of the vertebrae to be segmented
    :param seg: output mask - NumPy Array
    :return: distance map - NumPy array
    """
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask + distance(posmask) * posmask

    return res


def normalize_vol_std(vol):
    """
    Z-Score normalization of volume

    :param vol: volume - NumPy array
    :return: normalized volume - NumPy array
    """
    vol_mean = np.mean(vol.flatten())
    vol_std = np.std(vol.flatten())

    return (vol - vol_mean) / vol_std


def normalize_vol_max_min(vol, max_dt, min_dt):
    """
    Normalize volume to range [-1, 1]

    :param vol: volume - NumPy array
    :param max_dt: max value of volume or dataset - float
    :param min_dt: min value of volume or dataset - float
    :return: normalized volume - NumPy array
    """
    center_range = (min_dt + max_dt) / 2
    range_values = (max_dt - min_dt) / 2

    vol = (vol - center_range) / range_values

    return vol


def padding_up_down_ones(vol, mask):
    """
    Padd volume up and down

    :param vol: volume - NumPy array
    :param mask: output mask - NumPy array
    :return: padded volume and mask - NumPy Array
    """
    vol1 = np.pad(vol, ((0, 0), (0, 0), (128, 128)), constant_values=((-1, -1), (-1, -1), (-1, -1)))
    mask1 = np.pad(mask, ((0, 0), (0, 0), (128, 128)))

    return vol1, mask1


def check_padding_ones(vol, mask):
    """
    Padd volume up and down if volume size smaller than 256x256x256

    :param vol: volume - NumPy array
    :param mask: output mask - NumPy array
    :return: padded volume and mask - NumPy Array
    """
    patch_size = np.array((256, 256, 256))
    padd = patch_size - np.array(vol.shape)

    padd[padd < 0] = 0
    padd = padd / 2

    vol1 = np.pad(vol, ((math.ceil(padd[0]), math.ceil(padd[0])), (math.ceil(padd[1]), math.ceil(padd[1])),
                        (math.ceil(padd[2]), math.ceil(padd[2]))), constant_values=((-1, -1), (-1, -1), (-1, -1)))
    mask1 = np.pad(mask, ((math.ceil(padd[0]), math.ceil(padd[0])), (math.ceil(padd[1]), math.ceil(padd[1])),
                          (math.ceil(padd[2]), math.ceil(padd[2]))))

    return vol1, mask1


def rand_mul_shi_vox(vol):
    """
    Multiplies the volume and shifts the volume image

    :param vol: volume - NumPy array
    :return: volume (with shifted and multiplied histogram) - NumPy array
    """

    mult = np.random.uniform(high=1.25, low=0.75)
    vox_shift = np.random.uniform(low=-0.25, high=0.25)
    vol = vol * mult
    vol = vol + vox_shift

    return vol


def flip_vol(vol, memory, mask):
    """
    Flip left-right

    :param vol: volume - NumPy array
    :param memory: memory mask - NumPy array
    :param mask: output mask - NumPy array
    :return: left-right fliped version of the volume, memory and mask - NumPy array
    """

    vol = np.flip(vol, 1)
    memory = np.flip(memory, 1)
    mask = np.flip(mask, 1)

    return vol, memory, mask


def zoom_z(vol, memory, mask):
    """
    Zoom volume in 3 axis

    :param vol: volume - NumPy array
    :param memory: memory mask - NumPy array
    :param mask: output mask - NumPy array
    :return: zoomed version of volume, memory and mask - NumPy array
    """

    zoom_P = np.random.uniform(low=0.8, high=1.2)
    zoom_R = np.random.uniform(low=0.8, high=1.2)
    zoom_I = np.random.uniform(low=0.7, high=1.3)

    vol = zoom(vol, (zoom_P, zoom_R, zoom_I), order=3, mode='nearest')
    memory = zoom(memory, (zoom_P, zoom_R, zoom_I), order=0)
    mask = zoom(mask, (zoom_P, zoom_R, zoom_I), order=0)

    return vol, memory, mask


def rotate3D(vol, memory, mask):
    """
    Rotate volume in 3 axis by angles between -20 and 20 degrees
    :param vol: volume - NumPy array
    :param memory: memory mask - NumPy array
    :param mask: output mask - NumPy array
    :return: rotated versions of volume, memory and mask - NumPy array
    """

    rot_P = np.random.uniform(low=-20, high=20)
    rot_R = np.random.uniform(low=-20, high=20)
    rot_I = np.random.uniform(low=-20, high=20)

    vol = rotate(vol, rot_R, axes=(0, 2), order=3, mode='nearest')
    vol = rotate(vol, rot_P, axes=(1, 2), order=3, mode='nearest')
    vol = rotate(vol, rot_I, axes=(0, 1), order=3, mode='nearest')

    memory = rotate(memory, rot_R, axes=(0, 2), order=0)
    memory = rotate(memory, rot_P, axes=(1, 2), order=0)
    memory = rotate(memory, rot_I, axes=(0, 1), order=0)

    mask = rotate(mask, rot_R, axes=(0, 2), order=0)
    mask = rotate(mask, rot_P, axes=(1, 2), order=0)
    mask = rotate(mask, rot_I, axes=(0, 1), order=0)

    return vol, memory, mask


def gauss_noise(vol):
    """
    Add gaussian noise to volume

    :param vol: volume - NumPy array
    :return: noisy version of the volume - NumPy array
    """

    noise = np.random.normal(loc=0.0, scale=np.random.uniform(low=0.001, high=0.03), size=vol.shape)
    vol = vol + noise

    return vol


def gauss_blur(vol):
    """
    Apply gaussian filter to volume

    :param vol: volume - NumPy array
    :return: filtered version of the volume - NumPy array
    """

    sigma = np.random.uniform(low=0.5, high=1.5)
    truncKernel = np.random.randint(low=3, high=7)

    vol = gaussian_filter(vol, sigma, truncate=truncKernel)

    return vol


def clean_memory(memory):
    """
    Clean memory

    :param memory: memory mask - NumPy array
    :return: all-zero memory mask - NumPy array
    """

    memory = np.zeros(memory.shape)

    return memory


def roll_imgs(vol, memory, mask, slice_bef, nb_1s):
    """
    Apply translation to images

    :param vol: volume - NumPy array
    :param memory: memory mask - NumPy array
    :param mask: output mask - NumPy array
    :param slice_bef: coordinates where to start cropping - NumPy array
    :param nb_1s: number of voxels == 1 at output mask - NumPy array
    :return: translated volume, memory, mask and new slice_bef - NumPy array
    """

    check_1s = 0

    if np.sum(memory) == 0:

        slice_bef[2] = np.amin(np.where(mask == 1)[2]) - 2

        while check_1s < nb_1s / 3:
            roll_P = slice_bef[0] + np.random.randint(low=-64, high=64)
            roll_R = slice_bef[1] + np.random.randint(low=-64, high=64)
            roll_I = slice_bef[2] + np.random.randint(low=0, high=64)

            check_1s = np.where(mask[roll_P:roll_P + 128, roll_R:roll_R + 128, roll_I:roll_I + 128] == 1)[0].size

        slice_bef[0] = roll_P
        slice_bef[1] = roll_R
        slice_bef[2] = roll_I

        if slice_bef[0] + 128 > vol.shape[0]:
            pad_value = slice_bef[0] + 128 - vol.shape[0] + 1
            vol = np.pad(vol, ((0, pad_value), (0, 0), (0, 0)), constant_values=((-1, -1), (-1, -1), (-1, -1)))
            memory = np.pad(memory, ((0, pad_value), (0, 0), (0, 0)))
            mask = np.pad(mask, ((0, pad_value), (0, 0), (0, 0)))

        if slice_bef[1] + 128 > vol.shape[1]:
            pad_value = slice_bef[1] + 128 - vol.shape[1] + 1
            vol = np.pad(vol, ((0, 0), (0, pad_value), (0, 0)), constant_values=((-1, -1), (-1, -1), (-1, -1)))
            memory = np.pad(memory, ((0, 0), (0, pad_value), (0, 0)))
            mask = np.pad(mask, ((0, 0), (0, pad_value), (0, 0)))

        if slice_bef[2] + 128 > vol.shape[2]:
            pad_value = slice_bef[2] + 128 - vol.shape[2] + 1
            vol = np.pad(vol, ((0, 0), (0, 0), (0, pad_value)), constant_values=((-1, -1), (-1, -1), (-1, -1)))
            memory = np.pad(memory, ((0, 0), (0, 0), (0, pad_value)))
            mask = np.pad(mask, ((0, 0), (0, 0), (0, pad_value)))

    else:

        while check_1s < nb_1s / 3:
            roll_P = slice_bef[0] + np.random.randint(low=-32, high=32)
            roll_R = slice_bef[1] + np.random.randint(low=-32, high=32)
            roll_I = slice_bef[2] + np.random.randint(low=-32, high=32)

            check_1s = np.where(mask[roll_P:roll_P + 128, roll_R:roll_R + 128, roll_I:roll_I + 128] == 1)[0].size

        slice_bef[0] = roll_P
        slice_bef[1] = roll_R
        slice_bef[2] = roll_I

        if slice_bef[0] + 128 > vol.shape[0]:
            pad_value = slice_bef[0] + 128 - vol.shape[0] + 1
            vol = np.pad(vol, ((0, pad_value), (0, 0), (0, 0)), constant_values=((-1, -1), (-1, -1), (-1, -1)))
            memory = np.pad(memory, ((0, pad_value), (0, 0), (0, 0)))
            mask = np.pad(mask, ((0, pad_value), (0, 0), (0, 0)))

        if slice_bef[1] + 128 > vol.shape[1]:
            pad_value = slice_bef[1] + 128 - vol.shape[1] + 1
            vol = np.pad(vol, ((0, 0), (0, pad_value), (0, 0)), constant_values=((-1, -1), (-1, -1), (-1, -1)))
            memory = np.pad(memory, ((0, 0), (0, pad_value), (0, 0)))
            mask = np.pad(mask, ((0, 0), (0, pad_value), (0, 0)))

        if slice_bef[2] + 128 > vol.shape[2]:
            pad_value = slice_bef[2] + 128 - vol.shape[2] + 1
            vol = np.pad(vol, ((0, 0), (0, 0), (0, pad_value)), constant_values=((-1, -1), (-1, -1), (-1, -1)))
            memory = np.pad(memory, ((0, 0), (0, 0), (0, pad_value)))
            mask = np.pad(mask, ((0, 0), (0, 0), (0, pad_value)))

    vol = vol[slice_bef[0]:slice_bef[0] + 128, slice_bef[1]:slice_bef[1] + 128, slice_bef[2]:slice_bef[2] + 128]
    vol = np.where(vol > 1, 1, vol)
    vol = np.where(vol < -1, -1, vol)
    memory = memory[slice_bef[0]:slice_bef[0] + 128, slice_bef[1]:slice_bef[1] + 128, slice_bef[2]:slice_bef[2] + 128]
    mask = mask[slice_bef[0]:slice_bef[0] + 128, slice_bef[1]:slice_bef[1] + 128, slice_bef[2]:slice_bef[2] + 128]

    return vol, memory, mask

