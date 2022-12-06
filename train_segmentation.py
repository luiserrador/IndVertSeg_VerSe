import tensorflow as tf
from utils.train_segmentation_utils import DatasetHandler


if __name__ == '__main__':

    training_directory = 'D:/Datasets/VerSe19-20/Iso_Croped/Training/'
    validation_directory = 'D:/Datasets/VerSe19-20/Iso_Croped/Validation/'
    training_size = 2814
    validation_size = 1494
    batch_size = 1 * tf.distribute.get_strategy().num_replicas_in_sync
    datasets = DatasetHandler(training_directory, validation_directory, training_size, validation_size, batch_size)
    train_data, valid_data = datasets.get_datasets()
