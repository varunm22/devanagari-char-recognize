from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization


def alexnet_bn():
    model = Sequential()

    model.add(Conv2D(96, (5, 5), activation='relu', input_shape=(128, 128, 1)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2,2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2,2)))

    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(46, activation='softmax'))

    print('AlexNet model created.')
    return model
