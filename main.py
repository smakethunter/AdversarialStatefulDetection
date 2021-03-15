import tensorflow as tf
import matplotlib.pyplot as plt
from noises import *
# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
from tensorflow.python.compiler.mlcompute import mlcompute

# Select CPU device.
mlcompute.set_mlc_device(device_name='cpu') # Available options are 'cpu', 'gpu', and ‘any'.
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

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['accuracy'],
)

history = model.fit(
    train_data_noisy[:4000],train_data_clean[:4000],

    epochs=12,
    validation_split=0.2,

)
plt.plot(history.history['loss'])
plt.show()

