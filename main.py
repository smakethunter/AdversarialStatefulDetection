import tensorflow as tf
import matplotlib.pyplot as plt
from noises import *


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')




def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

batch_size = 128
from tensorflow.keras.datasets import cifar100, cifar10



(train_data_clean, _), (test_data_clean, _) = cifar100.load_data()
train_data_clean = train_data_clean.astype('float32')/255.
train_data_noisy = np.zeros(train_data_clean.shape)
for i in range(train_data_clean.shape[0]):
    train_data_noisy[i] =  normal_noise(train_data_clean[i])
test_data_noisy = np.zeros(test_data_clean.shape)
for i in range(test_data_clean.shape[0]):
    test_data_noisy[i] =  normal_noise(test_data_clean[i])

# ds_train = ds_train.map(
#     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# ds_train = ds_train.batch(batch_size)
# ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
#
#
# ds_test = ds_test.map(
#     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_test = ds_test.batch(batch_size)
# ds_test = ds_test.cache()
# ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

from encoders import *
model = UNetModel(32,256)
# tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
#                  activation='relu'),
#   tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
#                  activation='relu'),
# #   tf.keras.layers.Dropout(0.25),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Reshape((8,8,2)),
# #   tf.keras.layers.Dropout(0.5),
#   tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2,
#                  activation='relu', padding='same'),
#   tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2,
#                  activation='relu', padding='same'),
#   tf.keras.layers.Conv2DTranspose(3, kernel_size=(3, 3),
#                  activation='relu',padding='same'),
#
#
# ])
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(0.001, clipvalue=0.1, clipnorm=0.5),
    metrics=['accuracy'],
)

history = model.fit(
    train_data_noisy[:100],train_data_clean[:100],
    batch_size=4,
    epochs=12,
    validation_split=0.2
)
plt.plot(history.history['loss'])
plt.show()