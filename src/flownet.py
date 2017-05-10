from keras.models import Model, Sequential
from keras.layers import Activation, Input, Reshape, merge, Lambda, Dropout, Flatten, Dense
from keras.layers.merge import add
from keras.layers.convolutional import Convolution2D, Deconvolution2D, ZeroPadding2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
import keras.backend as K
import numpy as np
#from keras.utils.visualize_util import plot

def getModel(height = 384, width = 512):

    ## convolution model

    # left model
    input_l = Input(shape=(height, width, 3), name='pre_input')
    conv1_l = Convolution2D(64, (7, 7), padding = 'same', name='conv1_l')(input_l)
    conv1_l = MaxPooling2D(name='maxpool1_l')(conv1_l)
    conv2_l = Convolution2D(128, (5, 5), padding = 'same', name='conv2_l')(conv1_l)
    conv2_l = MaxPooling2D(name='maxpool2_l')(conv2_l)
    conv3_l = Convolution2D(256, (5, 5), padding = 'same', name='conv3_l')(conv2_l)
    conv3_l = MaxPooling2D(name='maxpool3_l')(conv3_l)

    # right model
    input_r = Input(shape=(height, width, 3), name='nxt_input')
    conv1_r = Convolution2D(64, (7, 7), padding = 'same', name='conv1_r')(input_r)
    conv1_r = MaxPooling2D(name='maxpool1_r')(conv1_r)
    conv2_r = Convolution2D(128, (5, 5), padding = 'same', name='conv2_r')(conv1_r)
    conv2_r = MaxPooling2D(name='maxpool2_r')(conv2_r)
    conv3_r = Convolution2D(256, (5, 5), padding = 'same', name='conv3_r')(conv2_r)
    conv3_r = MaxPooling2D(name='maxpool3_r')(conv3_r)

    # merge
    add_layer = add([conv3_l, conv3_r])

    # merged convolution
    conv3_1 = Convolution2D(256, (3, 3), padding = 'same', name='conv3_1')(add_layer)
    conv4 = Convolution2D(512, (3, 3), padding = 'same', name='conv4')(conv3_1)
    conv4 = MaxPooling2D(name='maxpool4')(conv4)
    conv4_1 = Convolution2D(512, (3, 3), padding = 'same', name='conv4_1')(conv4)
    conv5 = Convolution2D(512, (3, 3), padding = 'same', name='conv5')(conv4_1)
    conv5 = MaxPooling2D(name='maxpool5')(conv5)
    conv5_1 = Convolution2D(512, (3, 3), padding = 'same', name='conv5_1')(conv5)
    conv6 = Convolution2D(1024, (3, 3), padding = 'same', name='conv6')(conv5_1)
    conv6 = MaxPooling2D(name='maxpool6')(conv6)
    flatten_image = Flatten()(conv6)

    ## inertial model
    input_imu = Input(shape=(1, 6), name='imu_input')
    imu_lstm = LSTM(4, name='imu_lstm')(input_imu)

    ## core LSTM


    # whole model
    model = Model(inputs = [input_l, input_r], outputs = flatten_image)
    return model


if __name__ == '__main__':
    model = getModel()
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.summary()
    #model.fit({'pre_input': 0, 'nxt_input': 0}, {}, epochs=50, batch_size=32)
