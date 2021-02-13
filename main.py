import tensorflow as tf
import matplotlib.pyplot as plt
from noises import *


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='any')





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


from encoders import *
model = SimpleDAE(256)
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
    optimizer='adam',
    metrics=['accuracy'],
)

history = model.fit(
    train_data_noisy[:100],train_data_clean[:100],

    epochs=12,
    validation_split=0.2
)
plt.plot(history.history['loss'])
plt.show()

