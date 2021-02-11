import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU,Conv2D, BatchNormalization, Flatten, InputLayer, concatenate, ReLU
from tensorflow.keras.layers import Conv2DTranspose,Input,Dense,Reshape, Activation, Flatten, Concatenate
from tensorflow.keras.models import Sequential


def conv_block(filters, kernel_size=3, strides=2, padding='same'):
    x = Sequential([Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding),
                    BatchNormalization(),
                    ReLU()])
    return x


def deconv_block(filters, kernel_size=3, strides=2, padding='same'):
    x = Sequential([Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding),
                    BatchNormalization(),
                    ReLU()])
    return x


class UNetModel(tf.keras.Model):
    def __init__(self, filters, encoded_features, dense_layer=False):
        super().__init__()
        self.dense_layer = dense_layer
        self.conv1 = conv_block(filters, kernel_size=3, strides=2, padding='same')
        self.conv2 = conv_block(2 * filters, kernel_size=3, strides=2, padding='same')
        self.conv3 = conv_block(4 * filters, kernel_size=3, strides=2, padding='same')
        self.conv4 = conv_block(8 * filters, kernel_size=3, strides=2, padding='same')
        self.conv5 = conv_block(8 * filters, kernel_size=3, strides=1, padding='same')
        self.deconv1 = deconv_block(8 * filters, kernel_size=3, strides=2, padding='same')
        self.deconv2 = deconv_block(4 * filters, kernel_size=3, strides=2, padding='same')
        self.deconv3 = deconv_block(2 * filters, kernel_size=3, strides=2, padding='same')
        self.deconv4 = deconv_block(filters, kernel_size=3, strides=2, padding='same')
        self.deconv5 = deconv_block(3, kernel_size=3, strides=1, padding='same')
        self.dense_edncoder = Dense(encoded_features, activation='relu')
        self.dense_decoder = Dense(4 * 256, activation='relu')
        self.reshape = Reshape((2, 2, 256))

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = Flatten()(x)
        if self.dense_layer:
            x = self.dense_edncoder(x)
        return x
    @tf.function
    def call(self, x):
        conv_block1 = self.conv1(x)
        conv_block2 = self.conv2(conv_block1)
        conv_block3 = self.conv3(conv_block2)
        conv_block4 = self.conv4(conv_block3)
        d = self.conv5(conv_block4)
        if self.dense_layer:
            d = Flatten()(d)
            d = self.dense_edncoder(d)
            d = self.dense_decoder(d)
            d = self.reshape(d)
        deconv_block1 = self.deconv1(d)
        merge1 = Concatenate()([deconv_block1, conv_block3])
        deconv_block2 = self.deconv2(merge1)
        merge2 = Concatenate()([deconv_block2, conv_block2])
        deconv_block3 = self.deconv3(merge2)
        merge3 = Concatenate()([deconv_block3, conv_block1])
        deconv_block4 = self.deconv4(merge3)

        final_deconv = self.deconv5(deconv_block4)

        dae_outputs = Activation('sigmoid', name='dae_output')(final_deconv)
        return dae_outputs

    pass


def build_encoder(encoded_dimension):
    model = Sequential([

        Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(encoded_dimension, activation='relu')
    ])
    return model


def build_decoder():
    model = Sequential([
        Dense(16 * 16 * 256, activation='relu'),
        Reshape((16, 16, 256)),

        Conv2DTranspose(filters=256, kernel_size=3, strides=1, padding='valid'),
        ReLU(),
        BatchNormalization(),
        Conv2DTranspose(filters=256, kernel_size=3, strides=1, padding='valid'),
        ReLU(),
        BatchNormalization(),
        Conv2DTranspose(filters=128, kernel_size=3, strides=1, padding='valid'),
        ReLU(),
        BatchNormalization(),
        Conv2DTranspose(filters=128, kernel_size=3, strides=1, padding='valid'),
        ReLU(),
        BatchNormalization(),
        Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='valid'),
        ReLU(),
        BatchNormalization(),
        Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='valid'),
        ReLU(),
        BatchNormalization(),
        Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='valid'),
        ReLU(),
        BatchNormalization(),
        Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='valid'),
        Activation('sigmoid')
    ])
    return model


class SimpleDAE(tf.keras.Model):
    def __init__(self, encoded_dimension):
        super().__init__()

        self.encoder = build_encoder(encoded_dimension)
        self.decoder = build_decoder()

    @tf.function
    def call(self, x):
        x = self.encoder(x)
        return self.decoder(x)