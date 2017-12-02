import os
import shutil
import numpy as np
import math
import glob
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Dense, Flatten, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist


def build_decoder(inter_dim=256, input_dim=1024):
    def layer(inputs):
        decoder_h = Dense(inter_dim, activation='relu')(inputs)
        return Dense(input_dim, activation='sigmoid')(decoder_h)
    return layer


def build_vae(input_shape):
    '''
    Builds VAE model that consists of a simple 2 hidden-layer fully connected NN
    for both the encoder and decoder networks. Latent variable representation use
    256 variables for a 32x32 (1024-dimensional) input image.

    '''
    # Hyperparameters
    batch_size = 100
    inter_dim = 256
    latent_dim = 2
    input_dim = input_shape[0]*input_shape[1]
    epsilon_std = 1.0

    # Encoder network
    x = Input(shape=(input_dim, ))
    h = Dense(inter_dim, activation='relu')(x)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # Sample latent variables from distribution defined by z_mean and z_log_var
    def sample_encodings(args):
        z_mean, z_log_sigma = args

        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_sigma) * epsilon

    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            xent_loss = input_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            return x

    z = Lambda(sample_encodings, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder network
    x_decoded_mean = build_decoder()(z)
    # decoder_h = Dense(inter_dim, activation='relu')(z)
    # x_decoded_mean = Dense(input_dim, activation='sigmoid')(decoder_h)
    loss = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, loss)
    return vae

def get_images(dir_name):
    image_stack = []
    for img in glob.glob('../Data/%s/*/*.png' % dir_name): # All png images
        image_stack.append(cv2.imread(img,0))
    return np.stack(image_stack, axis=0)

    
def get_generators(batch_size):
    '''
     Produces Keras ImageDataGenerators for the training and validation datasets
     Generators can be customized to include various image augmentation functions
     and to different datasizes.

     Currently generates grayscale images of size 32x32 (as provided in the dataset)
    '''

    train_datagen = ImageDataGenerator(
        rescale=1./255
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        '../Data/Train',
        target_size=(32, 32),
        color_mode='grayscale',
        class_mode=None,
        batch_size=batch_size
    )

    validation_generator = validation_datagen.flow_from_directory(
        '../Data/Val',
        target_size=(32, 32),
        color_mode='grayscale',
        class_mode=None,
        batch_size=batch_size
    )

    return train_generator, validation_generator

if __name__ == "__main__":
    x_train = get_images("Train")
    x_val = get_images("Val")

    # Rescale pixel values
    x_train = x_train.astype('float32') / 255.
    x_val = x_val.astype('float32') / 255.

    # Flatten images into column vectors
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))

    batch_size = 100
    img_dim = 32
    opt = Adam(lr=1e-3)

    input_shape = (img_dim, img_dim, 1)

    vae_model = build_vae(input_shape)
    vae_model.compile(optimizer=opt, loss=None)

    vae_model.fit(x_train,
        shuffle=True,
        epochs=25,
        batch_size=batch_size,
        validation_data=(x_val, None))

