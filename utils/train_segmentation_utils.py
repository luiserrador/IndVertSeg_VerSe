import os
import nibabel as nib
import tensorflow as tf
import time

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
        train_dist_ds = tf.distribute.get_strategy().experimental_distribute_dataset(self.getTrainingDataset())
        # train_dist_ds = get_training_dataset(fold_train_filenames)

        # Hitting End Of Dataset exceptions is a problem in this setup. Using a repeated validation set instead.
        # This will introduce a slight inaccuracy because the validation dataset now has some repeated elements.
        valid_dist_ds = tf.distribute.get_strategy().experimental_distribute_dataset(self.getValidDataset())
        # valid_dist_ds = get_validation_dataset(fold_valid_filenames, repeated=True)

        train_data_iter = iter(train_dist_ds)  # the training data iterator is repeated and it is not reset
        # for each validation run (same as model.fit)

        valid_data_iter = iter(valid_dist_ds)  # the validation data iterator is repeated and it is not reset
        # for each validation run (different from model.fit where the
        # recommendation is to use a non-repeating validation dataset)

        return train_data_iter, valid_data_iter


class Trainer:
    """ Generic trainer

    Parameters
    -----------
    model : Tensorflow Model
        Model to train
    optimizer : Tensorflow Optimizer
        Optimizer to use
    learning_rate : float
        Learning rate
    model_dir : str
        Directory where to save the checkpoints / model
    """

    def __init__(self, model, optimizer, learning_rate, model_dir):

        self.model = model

        with tf.distribute.get_strategy().scope():
            self.train_accuracy = tf.keras.metrics.Sum()
            self.valid_accuracy = tf.keras.metrics.Sum()
            self.train_loss = tf.keras.metrics.Sum()
            self.valid_loss = tf.keras.metrics.Sum()

            self.optimizer = optimizer(learning_rate=learning_rate)

            ckpt = tf.train.Checkpoint(optimizer=self.optimizer, net=self.model)
            self.manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
            ckpt.restore(self.manager.latest_checkpoint)

        check_dir = os.path.exists(model_dir)
        if not check_dir:
            os.makedirs(model_dir)

        self.step_dir = os.path.join(model_dir, "step.npy")
        self.model_dir = model_dir

        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
            self.step = np.load(self.step_dir)
        else:
            print("Initializing from scratch.")
            self.step = 0

    def train(self, train_ds, valid_ds, train_size, validation_size, BATCH_SIZE, EPOCHS, save_step=1):
        """ Train the model

        Parameters
        -----------
        train_ds : tf.data.Dataset
            Training dataset
        valid_ds : tf.data.Dataset
            Validation dataset
        train_size : scalar
            Size of the training dataset
        validation_size : scalar
            Size of the validation dataset
        loss_fn : function
            Loss function
        accuracy_fn : function
            Accuracy function
        BATCH_SIZE : int
            Batch size
        EPOCHS : int
            Number of epochs to train
        """

        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

        with tf.distribute.get_strategy().scope():

            self.loss_fn = self.weight_loss_bound
            self.accuracy_fn = self.dice_hard_coe

        self.STEPS_PER_CALL = STEPS_PER_EPOCH = train_size // self.BATCH_SIZE
        self.VALIDATION_STEPS_PER_CALL = validation_size // self.BATCH_SIZE
        self.epoch = self.step // STEPS_PER_EPOCH
        epoch_steps = 0
        epoch_start_time = time.time()

        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

        train_data_iter = iter(train_ds)
        valid_data_iter = iter(valid_ds)

        if self.epoch < self.EPOCHS:

            while True:
                # run training step
                print('\nEPOCH {:d}/{:d}'.format(self.epoch + 1, self.EPOCHS))
                self.train_step(train_data_iter)
                epoch_steps += self.STEPS_PER_CALL
                self.step += self.STEPS_PER_CALL
                print(epoch_steps, '/', STEPS_PER_EPOCH)

                # validation run at the end of each epoch
                if (self.step // STEPS_PER_EPOCH) > self.epoch:
                    # validation run
                    valid_epoch_steps = 0
                    self.valid_step(valid_data_iter)
                    valid_epoch_steps += self.VALIDATION_STEPS_PER_CALL

                    # compute metrics
                    history['acc'].append(self.train_accuracy.result().numpy() / (self.BATCH_SIZE * epoch_steps))
                    history['val_acc'].append(
                        self.valid_accuracy.result().numpy() / (self.BATCH_SIZE * valid_epoch_steps))
                    history['loss'].append(self.train_loss.result().numpy() / (self.BATCH_SIZE * epoch_steps))
                    history['val_loss'].append(self.valid_loss.result().numpy() / (self.BATCH_SIZE * valid_epoch_steps))

                    # report metrics
                    epoch_time = time.time() - epoch_start_time
                    print('time: {:0.1f}s'.format(epoch_time),
                          'loss: {:0.4f}'.format(history['loss'][-1]),
                          'acc: {:0.4f}'.format(history['acc'][-1]),
                          'val_loss: {:0.4f}'.format(history['val_loss'][-1]),
                          'val_acc: {:0.4f}'.format(history['val_acc'][-1]))

                    # save checkpoint and training_step
                    if save_step and self.epoch % save_step == 0:
                        model_path = (os.path.join(self.model_dir, 'model_epoch_%s.h5' % (self.epoch + 1)))
                    self.model.save(model_path)
                    self.manager.save()
                    np.save(self.step_dir, self.step)

                    # set up next epoch
                    self.epoch = self.step // STEPS_PER_EPOCH
                    epoch_steps = 0
                    epoch_start_time = time.time()
                    self.train_accuracy.reset_states()
                    self.valid_accuracy.reset_states()
                    self.valid_loss.reset_states()
                    self.train_loss.reset_states()
                    if self.epoch >= self.EPOCHS:
                        print('Training done, {} epochs'.format(self.epoch))
                        break
        else:
            print('\nAlready trained!')

    @tf.function
    def weight_loss_bound(self, y_true, y_pred):
        y_true_array = tf.reshape(y_true[:, :, :, :, 0], [self.BATCH_SIZE, 128, 128, 128, 1])
        y_true_dist_map = tf.reshape(y_true[:, :, :, :, 1], [self.BATCH_SIZE, 128, 128, 128, 1])
        power_2 = tf.fill(y_true_array.shape, 2.0)
        gama = tf.fill(y_true_array.shape, 8.0)
        delta = tf.fill(y_true_array.shape, 36.0)
        ones_matrix = tf.fill(y_true_array.shape, 1.0)
        weight = tf.add(tf.multiply(gama, tf.exp(tf.divide(-tf.pow(y_true_dist_map, power_2), delta))), ones_matrix)
        fp_soft = tf.reduce_sum(tf.multiply(tf.multiply(weight, (tf.subtract(ones_matrix, y_true_array))), y_pred))
        fn_soft = tf.reduce_sum(tf.multiply(tf.multiply(weight, y_true_array), tf.subtract(ones_matrix, y_pred)))
        Lambda_min = tf.constant([0.1], dtype=tf.float32)
        Lambda_max = tf.constant([1.0], dtype=tf.float32)
        decim = tf.constant([10.0], dtype=tf.float32)
        half = tf.constant([3.0], dtype=tf.float32)  # half=3, cause i'm training 30 more epochs than expected (60->90)
        delta = tf.divide(tf.subtract(tf.cast(self.epoch, tf.float32), tf.divide(self.EPOCHS, half)), tf.divide(self.EPOCHS, decim))
        Lambda = tf.add(Lambda_min, tf.divide(tf.subtract(Lambda_max, Lambda_min), tf.add(Lambda_max, tf.exp(
            tf.subtract(tf.constant([0.0], dtype=tf.float32), delta)))))
        loss = tf.add(tf.multiply(Lambda, fp_soft), fn_soft)

        return loss

    @tf.function
    def dice_hard_coe(self, y_true, y_pred):

        threshold = 0.5
        axis = (1, 2, 3)
        smooth = 1e-5
        y_true = tf.reshape(y_true[:, :, :, :, 0], [self.BATCH_SIZE, 128, 128, 128, 1])
        y_true = tf.cast(y_true > threshold, dtype=tf.float32)
        y_pred = tf.reshape(y_pred[:, :, :, :, 0], [self.BATCH_SIZE, 128, 128, 128, 1])
        y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
        inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=axis)
        l = tf.reduce_sum(y_pred, axis=axis)
        r = tf.reduce_sum(y_true, axis=axis)
        hard_dice = (2. * inse + smooth) / (l + r + smooth)
        ##
        hard_dice = tf.reduce_mean(hard_dice, name='hard_dice')

        return hard_dice

    @tf.function
    def train_step(self, data_iter):
        def train_step_fn(images, labels):
            with tf.GradientTape() as tape:
                probabilities = self.model(images, training=True)
                loss = self.loss_fn(tf.cast(labels, tf.float32), probabilities)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            accuracy = self.accuracy_fn(tf.cast(labels, tf.float32), probabilities)
            self.train_accuracy.update_state(accuracy)
            self.train_loss.update_state(loss)

        for _ in tf.range(self.STEPS_PER_CALL):
            tf.distribute.get_strategy().run(train_step_fn, next(data_iter))

    @tf.function
    def valid_step(self, data_iter):
        def valid_step_fn(images, labels):
            probabilities = self.model(images, training=False)
            loss = self.loss_fn(tf.cast(labels, tf.float32), probabilities)
            accuracy = self.accuracy_fn(tf.cast(labels, tf.float32), probabilities)
            self.valid_accuracy.update_state(accuracy)
            self.valid_loss.update_state(loss)

        for _ in tf.range(self.VALIDATION_STEPS_PER_CALL):
            tf.distribute.get_strategy().run(valid_step_fn, next(data_iter))
