import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Model, Sequential
from keras.layers import Activation, Input, Reshape, merge, Lambda, Dropout, Flatten, Dense,LSTM
from keras.layers.merge import add,concatenate,dot
from keras.layers.convolutional import Convolution2D, Deconvolution2D, ZeroPadding2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add,Concatenate,Dot
import keras.backend as K
import numpy as np
import tensorflow as tf
#from keras.utils.visualize_util import plot

from scipy import misc
from scipy.linalg import logm, expm

NUM_INST = 10

def myDot():
    return Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0],x[1]),axis=-1,keep_dims=True),name = 'myDot')

def get_padded_stride(b,displacement_x,displacement_y,height_8=384/8,width_8=512/8):
    slice_height = height_8- abs(displacement_y)
    slice_width = width_8 - abs(displacement_x)
    start_y = abs(displacement_y) if displacement_y < 0 else 0
    start_x = abs(displacement_x) if displacement_x < 0 else 0
    top_pad    = displacement_y if (displacement_y>0) else 0
    bottom_pad = start_y
    left_pad   = displacement_x if (displacement_x>0) else 0
    right_pad  = start_x
    
    gather_layer = Lambda(lambda x: tf.pad(tf.slice(x,begin=[0,start_y,start_x,0],size=[-1,slice_height,slice_width,-1]),paddings=[[0,0],[top_pad,bottom_pad],[left_pad,right_pad],[0,0]]),name='gather_{}_{}'.format(displacement_x,displacement_y))(b)
    return gather_layer

def get_correlation_layer(conv3_pool_l,conv3_pool_r,max_displacement=20,stride2=2,height_8=384/8,width_8=512/8):
    layer_list = []
    dotLayer = myDot()
    for i in range(-max_displacement, max_displacement+stride2,stride2):
        for j in range(-max_displacement, max_displacement+stride2,stride2):
            slice_b = get_padded_stride(conv3_pool_r,i,j,height_8,width_8)
            current_layer = dotLayer([conv3_pool_l,slice_b])
            layer_list.append(current_layer)
    return Lambda(lambda x: tf.concat(x, 3),name='441_output_concatenation')(layer_list)
    

def getModel(height = 384, width = 512):

    ## convolution model

    # left and model
    input_l = Input(shape=(height, width, 1), name='pre_input')
    input_r = Input(shape=(height, width, 1), name='nxt_input')
    #layer 1
    conv1 = Convolution2D(64,(7,7), padding = 'same', name = 'conv1')
    conv1_l = conv1(input_l)
    conv1_r = conv1(input_r)
    conv1_pool_l = MaxPooling2D(name='maxpool1_l')(conv1_l)
    conv1_pool_r = MaxPooling2D(name='maxpool1_r')(conv1_r)
    
    #layer 2
    conv2 = Convolution2D(128, (5, 5), padding = 'same', name='conv2')
    conv2_l = conv2(conv1_pool_l)
    conv2_r = conv2(conv1_pool_r)
    conv2_pool_l = MaxPooling2D(name='maxpool2_l')(conv2_l)
    conv2_pool_r = MaxPooling2D(name='maxpool2_r')(conv2_r)

    #layer 3
    conv3 = Convolution2D(256, (5, 5), padding = 'same', name='conv3_l')
    conv3_l = conv3(conv2_pool_l)
    conv3_r = conv3(conv2_pool_r)
    conv3_pool_l = MaxPooling2D(name='maxpool3_l')(conv3_l)
    conv3_pool_r = MaxPooling2D(name='maxpool3_r')(conv3_r)


    # merge
    add_layer = get_correlation_layer(conv3_pool_l, conv3_pool_r,max_displacement=20,stride2=2,height_8=height/8,width_8=width/8)

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
    core_lstm = concatenate([flatten_image, imu_lstm])
    core_lstm = Reshape((1, 97284))(core_lstm)
    core_lstm = LSTM(6, name='output')(core_lstm)

    # whole model
    model = Model(inputs = [input_l, input_r, input_imu], outputs = core_lstm)
    return model


def readData():
    img_folder = '../dat/test_data/image_00/data/'
    imu_folder = '../dat/test_data/oxts/data/'

    p_img_lst = []
    n_img_lst = []
    imu_lst = []

    for i in xrange(10):
        # read pre image
        p_img_file = '%s%010d.png' % (img_folder, i)
        p_img = misc.imread(p_img_file)
        p_img = np.expand_dims(p_img, axis=2)
        p_img_lst.append(p_img)
        
        # read nxt image
        n_img_file = '%s%010d.png' % (img_folder, i+1)
        n_img = misc.imread(n_img_file)
        n_img = np.expand_dims(n_img, axis=2)
        n_img_lst.append(n_img)
        
        # read imu
        imu_file = '%s%010d.txt' % (imu_folder, i)
        with open(imu_file) as f:
            line = f.readline()
            tmp = line.strip().split(' ')
            imu = np.array(tmp[11:14] + tmp[17:20])
            imu = np.expand_dims(imu, axis=0)
            imu_lst.append(imu)

    ans_file = '../dat/test_data/poses/pose'
    mat_lst = []
    with open(ans_file) as f:
        for line in f.readlines():
            ans = map(float, line.strip().split(' '))
            mat = np.reshape(ans + [0, 0, 0, 1], (4, 4))
            mat_lst.append(np.matrix(mat))

    ans_lst = []
    for i in range(NUM_INST):
        mat = np.linalg.inv(mat_lst[i]) * mat_lst[i+1]
        w = logm(mat[:3, :3])
        ans_lst.append([w[2,1], w[0,2], w[1,0]] + mat[:3, 3])
        
    p_img_lst = np.array(p_img_lst)
    n_img_lst = np.array(n_img_lst)
    imu_lst = np.array(imu_lst)
    ans_lst = np.array(ans_lst)
    return p_img_lst, n_img_lst, imu_lst, ans_lst


if __name__ == '__main__':
    model = getModel(height=375, width=1242)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.summary()

    p_img_lst, n_img_lst, imu_lst, ans_lst = readData()
    '''
    model.fit({'pre_input': p_img_lst, 'nxt_input': n_img_lst, 'imu_input': imu_lst}, \
            {'output': ans_lst}, epochs=1, batch_size=1)
    '''

