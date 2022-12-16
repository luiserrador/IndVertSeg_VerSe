"""pre_processing_heatmap_utils.py: Everything for processing and prepare VerSe dataset for spine location heatmap training."""

__author__ = "Luis Serrador"

import nibabel as nib
import os
import numpy as np
from utils.data_utilities import reorient_to, resample_nib, load_centroids, reorient_centroids_to, rescale_centroids, \
    save_centroids

import skimage.transform as skTrans


def ctd_iso_resamp(ctd_iso, img_shape):
    """
    Resample centroid coordinates to be according the new volume shape (64x64x128)
    :param ctd_iso: centroids
    :param img_shape: original volume shape
    :return: resample centroids
    """
    for j in range(1, len(ctd_iso)):
        ctd_iso[j][1] = ctd_iso[j][1] / img_shape[0] * 64
        ctd_iso[j][2] = ctd_iso[j][2] / img_shape[1] * 64
        ctd_iso[j][3] = ctd_iso[j][3] / img_shape[2] * 128

    return ctd_iso


def save_heatmap_data_training(training_raw, training_points):
    """
    Save training volumes and heatmaps for spine location
    :param training_raw: directorys of training volumes
    :param training_points: directorys of training centroids
    :return: None
    """

    save_path_ctd = 'Gaussian/Training/ctd/'
    save_path_vol = 'Gaussian/Training/vol_ctd/'

    if not os.path.exists(save_path_ctd):
        os.makedirs(save_path_ctd)
        os.makedirs(save_path_vol)

    save_path = 'Gaussian/Training/'

    n_files = len(training_raw)

    for file in range(n_files):

        if file + 1 < 10:
            nb = '00' + str(file + 1)

        elif file + 1 < 100:
            nb = '0' + str(file + 1)

        else:
            nb = str(file + 1)

        print('#File: ', nb)

        # load files
        img_nib = nib.load(training_raw[file])
        ctd_list = load_centroids(training_points[file])

        # Resample and Reorient data
        img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
        ctd_iso = rescale_centroids(ctd_list, img_nib, (1, 1, 1))

        img_iso = reorient_to(img_iso, axcodes_to=('P', 'R', 'I'))
        ctd_iso = reorient_centroids_to(ctd_iso, img_iso)

        # get voxel data
        im_np = img_iso.get_fdata()

        ctd_iso = ctd_iso_resamp(ctd_iso, im_np.shape)
        im_np = skTrans.resize(im_np, (64, 64, 128), order=3, preserve_range=True)

        save_centroids(ctd_iso, save_path + 'ctd/ctd' + nb + '.json')
        np.save(save_path + 'vol_ctd/vol' + nb + '.npy', im_np)


def save_heatmap_data_validation(valid_raw, valid_points):
    """
    Save validation volumes and heatmaps for spine location
    :param valid_raw: directorys of validation volumes
    :param valid_points: directorys of validation centroids
    :return: None
    """

    save_path_ctd = 'Gaussian/Validation/ctd'
    save_path_vol = 'Gaussian/Validation/vol_ctd'

    if not os.path.exists(save_path_ctd):
        os.makedirs(save_path_ctd)
        os.makedirs(save_path_vol)

    save_path = 'Gaussian/Validation/'

    n_files = len(valid_raw)

    for file in range(n_files):

        if file + 1 < 10:
            nb = '00' + str(file + 1)

        elif file + 1 < 100:
            nb = '0' + str(file + 1)

        else:
            nb = str(file + 1)

        print('#File: ', nb)

        # load files
        img_nib = nib.load(valid_raw[file])
        ctd_list = load_centroids(valid_points[file])

        # Resample and Reorient data
        img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
        ctd_iso = rescale_centroids(ctd_list, img_nib, (1, 1, 1))

        img_iso = reorient_to(img_iso, axcodes_to=('P', 'R', 'I'))
        ctd_iso = reorient_centroids_to(ctd_iso, img_iso)

        # get voxel data
        im_np = img_iso.get_fdata()

        ctd_iso = ctd_iso_resamp(ctd_iso, im_np.shape)
        im_np = skTrans.resize(im_np, (64, 64, 128), order=3, preserve_range=True)

        save_centroids(ctd_iso, save_path + 'ctd/ctd' + nb + '.json')
        np.save(save_path + 'vol_ctd/vol' + nb + '.npy', im_np)
