import os
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import math
from keras.optimizers import SGD, Adam
from keras import models
from keras import layers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from alexnet_bn import alexnet_bn
from resnet import ResnetBuilder

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

     Currently generates grayscale images of size 128x128 (as provided in the dataset)
    '''
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        '../Data/Train',
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size
    )

    validation_generator = validation_datagen.flow_from_directory(
        '../Data/Test',
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size
    )

    return train_generator, validation_generator

if __name__ == '__main__':
    batch_size = 64
    num_outputs = 46
    train_generator, validation_generator = get_generators(batch_size)
    
    # Alexnet
    model = alexnet_bn()

    # ResNet-18
    # model = ResnetBuilder.build_resnet_34((1,128,128), 46)

    def specific_schedule(epoch_num):
        lr_rate_list = [0.1 for i in range(15)]
        lr_rate_list.extend([0.01 for i in range(10)])
        if epoch_num < len(lr_rate_list):
            return lr_rate_list[epoch_num]
        else:
            return 0.001

    lrate = LearningRateScheduler(specific_schedule)
    optimizer = Adam(lr=1e-5)
    # optimizer = SGD(lr=0.1, momentum=0.9, decay=1e-4)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    checkpointer = ModelCheckpoint(filepath='models/alexnet_no_aug.h5', verbose=1, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=int(math.ceil(float(train_generator.samples) / batch_size)),
        epochs=25,
        callbacks=[checkpointer],
        validation_data=validation_generator,
        validation_steps=int(math.ceil(float(validation_generator.samples) / batch_size))
    )




