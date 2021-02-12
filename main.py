



import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU,Conv2D, BatchNormalization, Flatten, InputLayer, concatenate, ReLU
from tensorflow.keras.layers import Conv2DTranspose,Input,Dense,Reshape, Activation, Flatten, Concatenate
from tensorflow.keras.models import Sequential
from encoders import *
from noises import *
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100, cifar10

(train_data_clean, _), (test_data_clean, _) = cifar100.load_data()
train_data_clean = train_data_clean.astype('float32')/255.
train_data_noisy = np.zeros(train_data_clean.shape)
for i in range(train_data_clean.shape[0]):
    train_data_noisy[i] =  normal_noise(train_data_clean[i])
test_data_noisy = np.zeros(test_data_clean.shape)
for i in range(test_data_clean.shape[0]):
    test_data_noisy[i] =  normal_noise(test_data_clean[i])


import datetime
import tensorflow as tf
import os
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)


model = UNetModel(32,256*2*2)
model.compile(loss = 'mse', optimizer='adam', metrics=['accuracy'])

#padd2_history = model.fit(train_data_noisy[:100],train_data_clean[:100], batch_size=32,verbose=1, epochs=5, validation_split=0.15, callbacks=[callback])
