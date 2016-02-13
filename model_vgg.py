from __future__ import print_function

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras import backend as K


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_vgg_model():
    model = Sequential()
    model.add(Activation(activation=center_normalize, input_shape=(30, 64, 64)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', W_regularizer=l2(1e-3)))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', W_regularizer=l2(1e-3)))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='rmse')
    return model

