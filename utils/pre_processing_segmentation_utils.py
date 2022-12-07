import os
import numpy as np
import nibabel as nib

from utils.data_utils import normalize_vol_max_min, padding_up_down_ones, check_padding_ones, calc_centr_vertebras
from utils.data_utilities import reorient_to, resample_nib


def get_to_balance_array(training_derivatives):
    nb_vert = np.zeros(28)

    for j in range(len(training_derivatives)):

        print('file: ', j + 1)

        msk_nib = nib.load(training_derivatives[j])
        msk = msk_nib.get_fdata()

        for r in range(1, 29):

            if np.where(msk == r)[0].size > 0:
                nb_vert[r - 1] += 1

    mul_vert = np.amax(nb_vert) / nb_vert
    np.save('mul_vert.npy', mul_vert)
    return mul_vert


def get_array_data_training(training_derivatives, mul_vert=None):
    if not mul_vert:
        mul_vert = np.load('mul_vert.npy')

    arrayData = np.empty((0, 2), int)

    nb_vert = np.zeros(28)

    for j in range(len(training_derivatives)):

        print('file: ', j + 1)

        msk_nib = nib.load(training_derivatives[j])
        msk = msk_nib.get_fdata()

        for r in range(1, 26):

            if np.where(msk == r)[0].size > 0:

                mult = mul_vert[r - 1]

                while mult > 0:

                    save_bool = np.random.rand(1)

                    if save_bool <= mult:
                        arrayData = np.append(arrayData, np.array([[j, r]]), axis=0)

                        nb_vert[r - 1] += 1

                    mult -= 1

    np.save('arrayData_balanced.npy', arrayData)
    return arrayData


def save_iso_croped_data(training_raw, training_derivatives, arrayData=None):
    if not arrayData:
        arrayData = np.load('arrayData_balanced.npy')

    if not os.path.exists('Data/'):
        os.mkdir('Data/')

    array_img = np.empty(0)

    p_ans = [0, 0]

    i = -1

    slice_size = [256, 256, 256]

    for j in range(len(arrayData)):

        p = arrayData[j]
        if p[0] != p_ans[0] or p[1] != p_ans[1]:

            i += 1
            print('Image: ', i)

            img_nib = nib.load(training_raw[arrayData[j, 0]])
            msk_nib = nib.load(training_derivatives[arrayData[j, 0]])

            img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
            msk_iso = resample_nib(msk_nib, voxel_spacing=(1, 1, 1),
                                   order=0)  # or resample based on img: resample_mask_to(msk_nib, img_iso)

            img_iso = reorient_to(img_iso, axcodes_to=('P', 'R', 'I'))
            msk_iso = reorient_to(msk_iso, axcodes_to=('P', 'R', 'I'))

            # img_nib = nib.load(training_raw[arrayData[j, 0]])
            # msk_nib = nib.load(training_derivatives[arrayData[j, 0]])

            imgs_after_resamp = img_iso.get_fdata()
            mask_after_resamp = msk_iso.get_fdata()
            imgs_after_resamp = np.clip(imgs_after_resamp, -1024, 3071)

            imgs_after_resamp = normalize_vol_max_min(imgs_after_resamp, 3071, -1024)

            imgs_after_resamp, mask_after_resamp = padding_up_down_ones(imgs_after_resamp, mask_after_resamp)

            imgs_after_resamp, mask_after_resamp = check_padding_ones(imgs_after_resamp, mask_after_resamp)

            centr_vert = calc_centr_vertebras(mask_after_resamp, arrayData[j, 1])

            centr_vert = centr_vert - 128

            slice_bef = [64, 64, 64]

            for f in range(3):
                if centr_vert[f] + 256 >= imgs_after_resamp.shape[f]:
                    overplus = centr_vert[f] + 256 - imgs_after_resamp.shape[f]
                    centr_vert[f] = centr_vert[f] - overplus
                    slice_bef[f] = 64 - overplus
                elif centr_vert[f] < 0:
                    slice_bef[f] = 64 + centr_vert[f]
                    centr_vert[f] = 0

            imgs_after_resamp = imgs_after_resamp[centr_vert[0]:centr_vert[0] + slice_size[0],
                                centr_vert[1]:centr_vert[1] + slice_size[1],
                                centr_vert[2]:centr_vert[2] + slice_size[2]]

            mask_patch = mask_after_resamp[centr_vert[0]:centr_vert[0] + slice_size[0],
                         centr_vert[1]:centr_vert[1] + slice_size[1],
                         centr_vert[2]:centr_vert[2] + slice_size[2]]

            imgNifti = nib.Nifti1Image(imgs_after_resamp, affine=img_nib.affine)
            print('Img Size: ', imgNifti.header['dim'][1:4])
            mskNifti = nib.Nifti1Image(mask_patch, affine=msk_nib.affine)
            print('Mask Size: ', mskNifti.header['dim'][1:4])

            if i + 1 < 10:
                nb = '000' + str(i + 1)

            elif i + 1 < 100:
                nb = '00' + str(i + 1)

            elif i + 1 < 1000:
                nb = '0' + str(i + 1)

            else:
                nb = str(i + 1)

            nib.save(imgNifti, 'Data/Training/images/img' + nb + '.nii.gz')
            nib.save(mskNifti, 'Data/Training/masks/mask' + nb + '.nii.gz')

            array_img = np.append(array_img, (i, p[1]))
            p_ans = p

        else:
            array_img = np.append(array_img, (i, p[1]))

    array_img = np.reshape(array_img, (len(array_img) // 2, 2))

    np.save('Data/arrayToBalance.npy', array_img)

    print('# Saved Images: ', i)
    print('# Images Balanced: ', len(array_img))
