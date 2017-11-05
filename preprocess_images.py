import os
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import math
from keras.optimizers import SGD, Adam
from keras import models
from keras import layers
from alexnet_bn import alexnet_bn

def transfer_validation_data():
    '''
    Move 200 of the images from each class to a separate directory for
    validation purposes. This will allow us to create a separate 
    ImageDataGenerator for just the validation data.

    '''

    current_dir = os.path.dirname(os.path.realpath(__file__)) 
    data_dir = os.path.join(current_dir, 'Data')

    train_dir = os.path.join(data_dir, 'Train')
    val_dir = os.path.join(data_dir, 'Val')

    subdirectories = os.listdir(train_dir)

    for subdirectory in subdirectories:
        os.makedirs(os.path.join(val_dir, subdirectory))

    for subdirectory in subdirectories:
        destination = os.path.join(val_dir, subdirectory)
        current_subdir = os.path.join(train_dir, subdirectory)
        category_samples = os.listdir(current_subdir)
        images_to_move = category_samples[-200:]

        for image in images_to_move:
            shutil.move(os.path.join(current_subdir, image), destination)


def get_generators(batch_size):
    '''
     Produces Keras ImageDataGenerators for the training and validation datasets
     Generators can be customized to include various image augmentation functions
     and to different datasizes.

     Currently generates grayscale images of size 32x32 (as provided in the dataset)
    '''
    datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        'Data/Train',
        target_size=(32, 32),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size
    )

    validation_generator = validation_datagen.flow_from_directory(
        'Data/Val',
        target_size=(32, 32),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size
    )

    return train_generator, validation_generator

if __name__ == '__main__':
    batch_size = 64
    train_generator, validation_generator = get_generators(batch_size)
    model = alexnet_bn()
    optimizer = Adam(lr=1e-4)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=int(math.ceil(float(train_generator.samples) / batch_size)),
        epochs=25,
        callbacks=[],
        validation_data=validation_generator,
        validation_steps=int(math.ceil(float(train_generator.samples) / batch_size))
    )


