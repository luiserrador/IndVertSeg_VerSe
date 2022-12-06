import math
import tensorflow as tf
from pathlib import Path

from utils.data_heatmap_utils import *
from utils.data_utilities import load_centroids
from utils.data_utils import normalize_vol_max_min
from ML_3D_Unet.utils.layers_func import create_unet3d


def gaussian(xL, yL, zL, H, W, D, sigma=10):
    channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2 + (t - zL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in
               range(W) for t in range(D)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(H, W, D))

    return channel


class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):

        training_raw = Path('Gaussian/Training/vol_ctd/')
        training_ctd = Path('Gaussian/Training/ctd/')

        training_raw = [f for f in training_raw.resolve().glob('*.npy') if f.is_file()]
        training_ctd = [f for f in training_ctd.resolve().glob('*.json') if f.is_file()]

        max_dt = 3071
        min_dt = -1024

        ind_img = np.arange(num_samples)
        np.random.shuffle(ind_img)
        np.random.shuffle(ind_img)

        for r in range(num_samples):  # num_samples

            img = np.load(training_raw[ind_img[r]])
            img = np.clip(img, -1024, 3071)
            ctd_iso = load_centroids(training_ctd[ind_img[r]])

            heat = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

            for _, coordx, coordy, coordz in ctd_iso[1:]:
                heat += gaussian(coordx, coordy, coordz, 64, 64, 128)

            heat = heat / np.max(heat)

            ## -> Normalization [-1,1]
            img = normalize_vol_max_min(img, max_dt, min_dt)
            #             heat = normalize_vol_max_min(heat, np.amax(heat), 0)

            ## -> Augmentation
            check_1s = 0
            nb_1s = np.where(heat >= 0)[0].size
            while check_1s < nb_1s / 4:
                img, heat = augment_data(img, heat)
                check_1s = np.where(heat >= 0)[0].size
            img, heat = roll_imgs(img, heat)

            x = np.zeros((64, 64, 128, 1))
            x[:, :, :, 0] = img

            y = np.zeros((64, 64, 128, 2))
            y[:, :, :, 0] = heat
            y[:, :, :, 1] = np.ones((64, 64, 128)) - heat

            yield (x, y)

    def __new__(cls, num_samples=141):
        return tf.data.Dataset.from_generator(
            generator=cls._generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=((64, 64, 128, 1), (64, 64, 128, 2)),
            args=(num_samples,)
        )


class ArtificialDatasetValid(tf.data.Dataset):

    def _generator(num_samples):

        training_raw = Path('Gaussian/Validation/vol_ctd/')
        training_ctd = Path('Gaussian/Validation/ctd/')

        training_raw = [f for f in training_raw.resolve().glob('*.npy') if f.is_file()]
        training_ctd = [f for f in training_ctd.resolve().glob('*.json') if f.is_file()]

        max_dt = 3071
        min_dt = -1024

        #         for r in range(num_samples): #num_samples
        for r in range(num_samples):  # num_samples

            img = np.load(training_raw[r])
            img = np.clip(img, -1024, 3071)
            ctd_iso = load_centroids(training_ctd[r])

            heat = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

            for _, coordx, coordy, coordz in ctd_iso[1:]:
                heat += gaussian(coordx, coordy, coordz, 64, 64, 128)

            heat = heat / np.max(heat)

            ## -> Normalization [-1,1]
            img = normalize_vol_max_min(img, max_dt, min_dt)

            x = np.zeros((64, 64, 128, 1))
            x[:, :, :, 0] = img

            y = np.zeros((64, 64, 128, 2))
            y[:, :, :, 0] = heat
            y[:, :, :, 1] = np.ones((64, 64, 128)) - heat

            yield (x, y)

    def __new__(cls, num_samples=120):
        return tf.data.Dataset.from_generator(
            generator=cls._generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=((64, 64, 128, 1), (64, 64, 128, 2)),
            args=(num_samples,)
        )


def get_training_dataset(batch_size):
    dataset = ArtificialDataset().shuffle(40).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def get_valid_dataset(batch_size):
    dataset = ArtificialDatasetValid().batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def get_datasets(batch_size):
    training_dataset = tf.distribute.get_strategy().experimental_distribute_dataset(get_training_dataset(batch_size))
    valid_dataset = tf.distribute.get_strategy().experimental_distribute_dataset(get_valid_dataset(batch_size))

    return training_dataset, valid_dataset


def get_unet_heatmap():
    model = create_unet3d([64, 64, 128, 1], n_convs=2, n_filters=[8, 16, 32, 64], ksize=[3, 3, 3], padding='same',
                          activation='relu', pooling='max', norm='batch_norm', dropout=[0], depth=4,
                          upsampling=True)

    x = model.layers[-3].output
    x = tf.keras.layers.Conv3D(2, 1, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    new_model = tf.keras.models.Model(inputs=model.input, outputs=x)

    for _ in range(6):
        new_model._layers.pop(4)

    return new_model


def kl_dice(y_true, y_pred):
    kl = tf.keras.losses.KLDivergence()
    first_term = kl(y_true, y_pred)
    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)
    dice = (2. * inse) / (l + r)
    last_dice = 1 - dice

    loss = first_term + last_dice

    return loss
