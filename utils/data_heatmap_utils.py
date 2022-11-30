import numpy as np

from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter


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


def flip_vol(vol, heat):
    """
    Flip left-right

    :param vol: volume - NumPy array
    :param heat: heatmap - NumPy array
    :return: left-right fliped version of the volume and heatmap - NumPy array
    """

    vol = np.flip(vol, 1)
    heat = np.flip(heat, 1)

    return vol, heat


def zoom_z(vol, heat):
    """
    Zoom volume in 3 axis

    :param vol: volume - NumPy array
    :param heat: heatmap - NumPy array
    :return: zoomed version of volume and heatmap - NumPy array
    """

    zoom_P = np.random.uniform(low=0.9, high=1.1)
    zoom_R = np.random.uniform(low=0.9, high=1.1)
    zoom_I = np.random.uniform(low=0.9, high=1.1)

    vol = zoom(vol, (zoom_P, zoom_R, zoom_I), order=3)
    heat = zoom(heat, (zoom_P, zoom_R, zoom_I), order=3)

    padding = [[0, 0], [0, 0], [0, 0]]
    cut_vol = np.array([0, 64, 0, 64, 0, 128])
    size_vol = np.array([64, 64, 128])

    for dim_img in range(3):
        if vol.shape[dim_img] < size_vol[dim_img]:
            padding[dim_img] = [(size_vol[dim_img] - vol.shape[dim_img]) // 2 + 1,
                                (size_vol[dim_img] - vol.shape[dim_img]) // 2 + 1]
        elif vol.shape[dim_img] > size_vol[dim_img]:
            cut_vol[dim_img * 2] = vol.shape[dim_img] // 2 - size_vol[dim_img] // 2
            cut_vol[dim_img * 2 + 1] = vol.shape[dim_img] // 2 + size_vol[dim_img] // 2

    vol = np.pad(vol, padding, mode='constant', constant_values=((-1, -1), (-1, -1), (-1, -1)))
    heat = np.pad(heat, padding)

    vol = vol[cut_vol[0]:cut_vol[1], cut_vol[2]:cut_vol[3], cut_vol[4]:cut_vol[5]]
    heat = heat[cut_vol[0]:cut_vol[1], cut_vol[2]:cut_vol[3], cut_vol[4]:cut_vol[5]]

    return vol, heat


def rotate3D(vol, heat):
    """
    Rotate volume in 3 axis by angles between -10 and 10 degrees
    :param vol: volume - NumPy array
    :param heat: heatmap - NumPy array
    :return: rotated versions of volume and heatmap - NumPy array
    """

    rot_P = np.random.uniform(low=-10, high=10)
    rot_R = np.random.uniform(low=-10, high=10)
    rot_I = np.random.uniform(low=-10, high=10)

    vol = rotate(vol, rot_R, axes=(0, 2), order=3, reshape=False, cval=-1.0)
    vol = rotate(vol, rot_P, axes=(1, 2), order=3, reshape=False, cval=-1.0)
    vol = rotate(vol, rot_I, axes=(0, 1), order=3, reshape=False, cval=-1.0)

    heat = rotate(heat, rot_R, axes=(0, 2), order=3, reshape=False)
    heat = rotate(heat, rot_P, axes=(1, 2), order=3, reshape=False)
    heat = rotate(heat, rot_I, axes=(0, 1), order=3, reshape=False)

    return vol, heat


def gauss_noise(vol):
    """
    Add gaussian noise to volume

    :param vol: volume - NumPy array
    :return: noisy version of the volume - NumPy array
    """

    noise = np.random.normal(loc=0.0, scale=np.random.uniform(low=0.001, high=0.005), size=vol.shape)
    vol = vol + noise

    return vol


def gauss_blur(vol):
    """
    Apply gaussian filter to volume

    :param vol: volume - NumPy array
    :return: filtered version of the volume - NumPy array
    """

    sigma = np.random.uniform(low=0.5, high=1)
    truncKernel = np.random.randint(low=3, high=7)

    vol = gaussian_filter(vol, sigma, truncate=truncKernel)

    return vol


def roll_imgs(vol, heat):
    """
    Apply translation to images

    :param vol: volume - NumPy array
    :param heat: heatmap - NumPy array
    :return: translated volume and heatmap - NumPy array
    """

    vol = np.pad(vol, ((20, 20), (20, 20), (20, 20)), mode='constant', constant_values=((-1, -1), (-1, -1), (-1, -1)))
    heat = np.pad(heat, ((20, 20), (20, 20), (20, 20)))

    nb_1s = np.where(heat >= 0)[0].size
    check_1s = 0

    while check_1s < nb_1s / 6:
        rand_1 = np.random.randint(low=-20, high=20)
        rand_2 = np.random.randint(low=-20, high=20)
        rand_3 = np.random.randint(low=-20, high=20)

        slice_vol = np.array([20 + rand_1, 20 + rand_2, 20 + rand_3])

        check_1s = np.where(
            heat[slice_vol[0]:slice_vol[0] + 64, slice_vol[1]:slice_vol[1] + 64, slice_vol[2]:slice_vol[2] + 128] >= 0)[
            0].size

    vol = vol[slice_vol[0]:slice_vol[0] + 64, slice_vol[1]:slice_vol[1] + 64, slice_vol[2]:slice_vol[2] + 128]
    heat = heat[slice_vol[0]:slice_vol[0] + 64, slice_vol[1]:slice_vol[1] + 64, slice_vol[2]:slice_vol[2] + 128]

    return vol, heat


def augment_data(img, heat):
    """
    Apply data augmentation
    :param img: volume image - NumPy array
    :param heat: heatmap - NumPy array
    :return: augmented version of volume and heatmap - NumPy array
    """

    if np.random.uniform() > 0.5:
        img, heat = flip_vol(img, heat)
    img = rand_mul_shi_vox(img)
    img, heat = zoom_z(img, heat)
    img, heat = rotate3D(img, heat)
    if np.random.uniform() < 0.8:
        img = gauss_noise(img)
    if np.random.uniform() < 0.8:
        img = gauss_blur(img)

    return img, heat