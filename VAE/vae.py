import os
import shutil
import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Dense, Flatten, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics

class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        return x

def build_vae(input_shape):
	'''
	Builds VAE model that consists of a simple 2 hidden-layer fully connected NN
	for both the encoder and decoder networks. Latent variable representation use
	256 variables for a 32x32 (964-dimensional) input image.

	'''
	# Hyperparameters
	batch_size = 64
	latent_dim = 256

	# Encoder network
	x = Input(shape=input_shape)
	x_flat = Flatten(x)
	h = Dense(latent_dim, activation='relu')(x_flat)

	z_mean = Dense(latent_dim)(h)
	z_log_var = Dense(latent_dim)(h)

	# Sample latent variables from distribution defined by z_mean and z_log_var
	def sample_encodings(args):
		mean, log_sigma = args
		epsilon = K.random_normal(shape=(batch_size, latent_dim),
	                              mean=0., std=epsilon_std)
	    return z_mean + K.exp(z_log_sigma) * epsilon

	z = Lambda(sample_encodings, output_shape=(latent_dim,))([z_mean, z_log_var])

	# Decoder network
	x_decoded_mean = build_decoder()(z)
	loss = CustomVariationalLayer()([x_flat, x_decoded_mean])
	vae = Model(x, loss)
	return vae


def build_decoder(latent_dim=256, input_dim=784):
	def layer(inputs):
		decoder_h = Dense(latent_dim, activation='relu')(inputs)
		decoder_mean = Dense(input_dim, activation='sigmoid')(decoder_h)
	return layer


def get_generators(batch_size):
    '''
     Produces Keras ImageDataGenerators for the training and validation datasets
     Generators can be customized to include various image augmentation functions
     and to different datasizes.

     Currently generates grayscale images of size 128x128 (as provided in the dataset)
    '''


    train_datagen = ImageDataGenerator(
        rescale=1./255
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        'Data/Train',
        target_size=(32, 32),
        color_mode='grayscale',
        class_mode=None,
        batch_size=batch_size
    )

    validation_generator = validation_datagen.flow_from_directory(
        'Data/Test',
        target_size=(32, 32),
        color_mode='grayscale',
        class_mode=None,
        batch_size=batch_size
    )

    return train_generator, validation_generator

if __name__ == "__main__":
	batch_size = 64
	img_dim = 32
	input_shape = (img_dim, img_dim, 1)

	train_generator, validation_generator = get_generators(batch_size)
	opt = Adam(lr=1e-4)

	vae_model = build_vae(input_shape)
	vae_model.compile(optimizer=opt, loss=None)
	vae_model.fit_generator(
        train_generator,
        steps_per_epoch=int(math.ceil(float(train_generator.samples) / batch_size)),
        epochs=25,
        validation_data=validation_generator,
        validation_steps=int(math.ceil(float(validation_generator.samples) / batch_size))
    )

