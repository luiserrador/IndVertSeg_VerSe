import os
import nibabel as nib
import tensorflow as tf

from pathlib import Path
from utils.data_utils import *


class DatasetHandler:
    """ Handling VerSe dataset for vertebrae segmentation

    Parameters
    -----------
    training_directory : str
        Training data directory
    validation_directory : str
        Validation data directory
    training_size : float
        Size of training dataset
    validation_size : float
        Size of validation dataset
    batch_size : float
        Batch size
    """

    def __init__(self, training_directory, validation_directory, training_size, validation_size, batch_size):
        self.array_training_img = np.load(os.path.join(training_directory, 'arrayToBalance_int.npy'))

        self.training_raw = Path(os.path.join(training_directory, 'images/'))
        self.training_derivatives = Path(os.path.join(training_directory, 'masks/'))

        self.training_raw = [f for f in self.training_raw.resolve().glob('*.nii.gz') if f.is_file()]
        self.training_derivatives = [f for f in self.training_derivatives.resolve().glob('*.nii.gz') if f.is_file()]

        self.training_raw = sorted(self.training_raw)
        self.training_derivatives = sorted(self.training_derivatives)

        self.array_valid_img = np.load(os.path.join(validation_directory, 'arrayToBalance.npy'))

        self.valid_raw = Path(os.path.join(validation_directory, 'images/'))
        self.valid_derivatives = Path(os.path.join(validation_directory, 'masks/'))

        self.valid_raw = [f for f in self.valid_raw.resolve().glob('*.nii.gz') if f.is_file()]
        self.valid_derivatives = [f for f in self.valid_derivatives.resolve().glob('*.nii.gz') if f.is_file()]

        self.valid_raw = sorted(self.valid_raw)
        self.valid_derivatives = sorted(self.valid_derivatives)

        self.batch_size = batch_size

        self.training_size = training_size
        self.validation_size = validation_size

        self.AUTO = tf.data.experimental.AUTOTUNE

    def process_img_train(self, ind: float) -> np.array:

        img_vert = self.array_training_img[ind]

        patch_nifti = nib.load(self.training_raw[img_vert[0]])
        mask_patch_nifti = nib.load(self.training_derivatives[img_vert[0]])

        patch = patch_nifti.get_fdata()
        mask_patch = mask_patch_nifti.get_fdata()

        mask_save = np.where(mask_patch == img_vert[1], 1, 0)
        memory_save = np.where((mask_patch < img_vert[1]) & (mask_patch != 0), 1, 0)

        ## -> AUGMENTATION
        if np.random.uniform() > 0.5:
            patch, memory_save, mask_save = flip_vol(patch, memory_save, mask_save)
        patch = rand_mul_shi_vox(patch)
        patch, memory_save, mask_save = zoom_z(patch, memory_save, mask_save)
        patch, memory_save, mask_save = rotate3D(patch, memory_save, mask_save)
        if np.random.uniform() > 0.2:
            patch = gauss_noise(patch)
        if np.random.uniform() > 0.2:
            patch = gauss_blur(patch)
        if np.random.uniform() > 0.7:
            memory_save = clean_memory(memory_save)
        nb_1s = np.where(mask_save == 1)[0].size

        slice_bef = calc_centr_vertebras(mask_save, 1)
        slice_bef = slice_bef - 64

        for f in range(3):
            if slice_bef[f] + 128 >= mask_save.shape[f]:
                overplus = slice_bef[f] + 128 - mask_save.shape[f]
                slice_bef[f] = slice_bef[f] - overplus
            elif slice_bef[f] < 0:
                slice_bef[f] = 0

        patch, memory_save, mask_save = roll_imgs(patch, memory_save, mask_save, slice_bef, nb_1s)

        ## -> SLICE 128X128X128

        # patch = patch[slice_bef[0]:slice_bef[0] + 128, slice_bef[1]:slice_bef[1] + 128, slice_bef[2]:slice_bef[2] + 128]
        # patch = np.where(patch > 1, 1, patch)
        # patch = np.where(patch < -1, -1, patch)
        # memory_save = memory_save[slice_bef[0]:slice_bef[0] + 128, slice_bef[1]:slice_bef[1] + 128,
        #               slice_bef[2]:slice_bef[2] + 128]
        # mask_save = mask_save[slice_bef[0]:slice_bef[0] + 128, slice_bef[1]:slice_bef[1] + 128,
        #             slice_bef[2]:slice_bef[2] + 128]

        ## -> X = patch + memory

        x = np.zeros((128, 128, 128, 2))
        x[:, :, :, 0] = patch
        x[:, :, :, 1] = memory_save

        ## -> Y = mask + distance map

        dist = calc_dist_map(mask_save)
        y = np.zeros((128, 128, 128, 2))
        y[:, :, :, 0] = mask_save
        y[:, :, :, 1] = dist

        return x, y

    def get_img_train(self, i: tf.Tensor) -> np.array:
        i = i.numpy()  # Decoding from the EagerTensor object
        x, y = self.process_img_train(i)
        return x, y

    def getTrainingDataset(self) -> tf.data.Dataset:
        z = tf.range(self.training_size)

        dataset = tf.data.Dataset.from_generator(lambda: z, tf.int32)

        dataset = dataset.shuffle(buffer_size=len(z), reshuffle_each_iteration=True)

        dataset = dataset.map(lambda i: tf.py_function(func=self.get_img_train,
                                                       inp=[i],
                                                       Tout=[tf.float32,
                                                             tf.float32]
                                                       ),
                              num_parallel_calls=self.AUTO)

        dataset = dataset.batch(self.batch_size).repeat().prefetch(1)

        return dataset

    def process_img_valid(self, ind: float) -> np.array:

        img_vert = self.array_valid_img[ind]

        patch_nifti = nib.load(self.valid_raw[img_vert[0]])
        mask_patch_nifti = nib.load(self.valid_derivatives[img_vert[0]])

        patch = patch_nifti.get_fdata()
        mask_patch = mask_patch_nifti.get_fdata()

        mask_save = np.where(mask_patch == img_vert[1], 1, 0)
        memory_save = np.where((mask_patch < img_vert[1]) & (mask_patch != 0), 1, 0)

        slice_bef = calc_centr_vertebras(mask_save, 1)
        slice_bef = slice_bef - 64

        for f in range(3):
            if slice_bef[f] + 128 >= mask_save.shape[f]:
                overplus = slice_bef[f] + 128 - mask_save.shape[f]
                slice_bef[f] = slice_bef[f] - overplus
            elif slice_bef[f] < 0:
                slice_bef[f] = 0

        ## -> SLICE 128X128X128

        patch = patch[slice_bef[0]:slice_bef[0] + 128, slice_bef[1]:slice_bef[1] + 128, slice_bef[2]:slice_bef[2] + 128]
        patch = np.where(patch > 1, 1, patch)
        patch = np.where(patch < -1, -1, patch)
        memory_save = memory_save[slice_bef[0]:slice_bef[0] + 128, slice_bef[1]:slice_bef[1] + 128,
                      slice_bef[2]:slice_bef[2] + 128]
        mask_save = mask_save[slice_bef[0]:slice_bef[0] + 128, slice_bef[1]:slice_bef[1] + 128,
                    slice_bef[2]:slice_bef[2] + 128]

        ## -> X = patch + memory

        x = np.zeros((128, 128, 128, 2))
        x[:, :, :, 0] = patch
        x[:, :, :, 1] = memory_save

        ## -> Y = mask + distance map

        dist = calc_dist_map(mask_save)
        y = np.zeros((128, 128, 128, 2))
        y[:, :, :, 0] = mask_save
        y[:, :, :, 1] = dist

        return x, y

    def get_img_valid(self, i: tf.Tensor) -> np.array:
        i = i.numpy()  # Decoding from the EagerTensor object
        x, y = self.process_img_valid(i)
        return x, y

    def getValidDataset(self) -> tf.data.Dataset:
        z = tf.range(self.validation_size)

        dataset = tf.data.Dataset.from_generator(lambda: z, tf.int32)

        dataset = dataset.shuffle(buffer_size=len(z), reshuffle_each_iteration=True)

        dataset = dataset.map(lambda i: tf.py_function(func=self.get_img_valid,
                                                       inp=[i],
                                                       Tout=[tf.float32,
                                                             tf.float32]
                                                       ),
                              num_parallel_calls=self.AUTO)

        dataset = dataset.batch(self.batch_size).repeat().prefetch(1)

        return dataset

    def get_datasets(self):

        # distribute the dataset according to the strategy
        # train_dist_ds = tf.distribute.get_strategy().experimental_distribute_dataset(self.getTrainingDataset())
        train_dist_ds = self.getTrainingDataset()
        # train_dist_ds = get_training_dataset(fold_train_filenames)

        # Hitting End Of Dataset exceptions is a problem in this setup. Using a repeated validation set instead.
        # This will introduce a slight inaccuracy because the validation dataset now has some repeated elements.
        # valid_dist_ds = tf.distribute.get_strategy().experimental_distribute_dataset(self.getValidDataset())
        valid_dist_ds = self.getValidDataset()
        # valid_dist_ds = get_validation_dataset(fold_valid_filenames, repeated=True)

        train_data_iter = iter(train_dist_ds)  # the training data iterator is repeated and it is not reset
        # for each validation run (same as model.fit)

        valid_data_iter = iter(valid_dist_ds)  # the validation data iterator is repeated and it is not reset
        # for each validation run (different from model.fit where the
        # recommendation is to use a non-repeating validation dataset)

        return train_data_iter, valid_data_iter