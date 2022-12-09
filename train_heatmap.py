from ML_3D_Unet.utils.train import Trainer
from utils.train_heatmap_utils import *

train_size = 141
valid_size = 120
batch_size = 4 * tf.distribute.get_strategy().num_replicas_in_sync
train_data, valid_data = get_datasets(batch_size)

with tf.distribute.get_strategy().scope():
    model = get_unet_heatmap()

lr = 1e-3
optimizer = tf.keras.optimizers.Adam
model_dir = 'tf_ckpt_heatmap'
n_epochs = 300

print('\nTraining Heatmap Network')

trainer = Trainer(model, optimizer, lr, model_dir)
trainer.train(train_ds=train_data, valid_ds=valid_data, train_size=train_size, validation_size=valid_size,
              loss_fn=kl_dice, accuracy_fn=tf.keras.metrics.mean_squared_error, BATCH_SIZE=batch_size, EPOCHS=300,
              save_step=1)