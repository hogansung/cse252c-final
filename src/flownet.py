import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.engine.topology import Layer
from keras.models import Model, Sequential
from keras import activations
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
from keras.optimizers import SGD
import itertools
#from keras.utils.visualize_util import plot
import random
from scipy import misc
from scipy.linalg import logm, expm
import pandas as pd
import scipy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, \
    img_to_array, load_img
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NUM_INST = 10
QUICK_DEBUG = True
BATCH_SIZE = 3
num_epochs = 6
num_train_sets = 9
batch_size = BATCH_SIZE
use_SE3 = True;
stateful=True;
if use_SE3:
    num_targets = 4
else:
    num_targets = 2

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

def batch_se3_to_skew_param(batch_se3_vector):
    output = tf.slice(batch_se3_vector,[0,0],[-1,3],name='se3_to_skew_param')
    return output

def batch_se3_to_delta_position(batch_se3_vector):
    output = tf.slice(batch_se3_vector,[0,3],[-1,3],name='se3_to_delta_position')
    return output

def batchSkewParamToSkewMatrix(skewParam):
    ''' takes a (batch_size,3) set of skew parameters and returns
     (batch_size,3,3) set of skew matrices'''
    indices = tf.constant([[2,1], [0,2], [1,0],[1,2],[2,0],[0,1]])
    # concatenate on the negatve of each skew parameter for easy indexing
    batch_skew_params =  tf.concat([skewParam,-1*skewParam],1)
    #transpose the parameters for scattering, using slices along the last dimension (the batch dimension)
    updates = tf.transpose(batch_skew_params,[1,0])
    shape = tf.shape(tf.expand_dims(tf.transpose(skewParam),axis=0)+tf.constant([0.0,0.0,0.0],shape=(3,1,1)))
    scatter = tf.scatter_nd(indices, updates, shape)
    outputs = K.permute_dimensions(scatter,[2,0,1])
    return outputs

def batchSkewParamToSO3(skewParam):
    skewMatrix = batchSkewParamToSkewMatrix(skewParam)
    eye3 = tf.diag(tf.ones([3]))
    skewSumSquares = tf.reduce_sum(tf.square(skewParam),axis=1,keep_dims=True)
    skewNorm = tf.sqrt(skewSumSquares)
    cond = skewNorm > 1e-7
    sinTerm = tf.sin(skewNorm)
    cosSkew = tf.cos(skewNorm)
    cosTerm = 1.0 - cosSkew
    a = tf.expand_dims(tf.where(cond,sinTerm/skewNorm, cosSkew),axis=-1)
    b = tf.expand_dims(tf.where(cond,cosTerm/skewSumSquares,0.5*cosSkew),axis=-1)
    skewMatrixSquare = tf.matmul(skewMatrix,skewMatrix)
    result = eye3 + tf.multiply(a,skewMatrix) + tf.multiply(b,skewMatrixSquare)
    return result
    
def loss_SO3(y_true,y_pred):
    ''' assumes y_true is a batch_size x 3x3 matrix representing'''
    y_true_transpose = K.permute_dimensions(y_true,[0,2,1])
    small_world_errors = tf.matmul(y_true_transpose,y_pred)
    small_z = tf.slice(small_world_errors,[0,0,1],[-1,1,1])
    small_y = tf.slice(small_world_errors,[0,0,2],[-1,1,1])
    small_x = tf.slice(small_world_errors,[0,1,2],[-1,1,1])
    square_errors = K.square(K.stack([small_x,small_y,small_z]))
    loss = K.mean(square_errors)
    return loss

def loss_angle_SE3(y_true,y_pred):
    ''' assumes y_true is a batch_size x 4x4 matrix representing'''
    y_pred_inverse = tf.matrix_inverse(y_pred)
    small_world_errors = tf.matmul(y_pred_inverse,y_true)
    small_z = tf.slice(small_world_errors,[0,0,1],[-1,1,1])
    small_y = tf.slice(small_world_errors,[0,0,2],[-1,1,1])
    small_x = tf.slice(small_world_errors,[0,1,2],[-1,1,1])
    square_errors = K.square(K.stack([small_x,small_y,small_z]))
    loss = K.mean(square_errors)
    return loss

def loss_position_SE3(y_true,y_pred):
    ''' assumes y_true is a batch_size x 4x4 matrix representing the current SE3 transform from world reference frame to camera reference frame'''
    y_true_inverse = tf.matrix_inverse(y_true)
    y_pred_inverse = tf.matrix_inverse(y_pred)
    small_world_errors = y_pred_inverse-y_true_inverse;
    small_x = tf.slice(small_world_errors,[0,0,3],[-1,1,1])
    small_y = tf.slice(small_world_errors,[0,1,3],[-1,1,1])
    small_z = tf.slice(small_world_errors,[0,2,3],[-1,1,1])
    square_errors = K.square(K.stack([small_x,small_y,small_z]))
    loss = K.mean(square_errors)
    return loss
    
class SE3ExpansionLayer(Layer):
    ''' Converts a 3x4 SE3 matrix to its 4x4 canonical representation by appending [0,0,0,1] to the bottom '''
    def __init__(self, **kwargs):
        super(SE3ExpansionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # No trainable weights, no problems
        super(SE3ExpansionLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        batch_size = x.shape[0]
        num_cols = 4
        num_zeros = num_cols -1;
        batch_array_shape = tf.shape(tf.reduce_sum(x,axis=1,keep_dims=True))
        zero_zero_zero_one = tf.zeros(shape=batch_array_shape)+tf.constant(np.array([0.0]*int(num_zeros)+[1.0]),dtype=K.floatx(),shape=(1,1,num_cols))
        output_tensor = tf.concat([x,zero_zero_zero_one],axis=1)
        return output_tensor

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1]+1,input_shape[2])
    
        
class SE3AccumulationLayer(Layer):
    '''Accumulates SE3 matrices based on self.initial_SE3
      Assumes inputs are size (batch_size,4,4)
      retunrs (batch_size,3,3)
      '''
    def __init__(self, initial_SE3 = None,stateful=False,batch_size=None, **kwargs):
        '''initial_SE3 is either a tensor if stateful==False or a list of tensors if stateful==True'''
        self.stateful = stateful
        self.batch_size = batch_size
        if not stateful:
            if initial_SE3 is None:
                initial_SE3 = K.eye(4)
            self.initial_SE3_init = initial_SE3
        else:
            if initial_SE3 is None:
                initial_SE3=[]
                assert batch_size is not None
                assert batch_size > 0
                for i in range(batch_size):
                    initial_SE3.append(K.eye(4))
            self.initial_SE3_init = initial_SE3
        super(SE3AccumulationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # No trainable weights, no problems
        self.initial_SE3 = []
        if self.stateful:
            for index,init in enumerate(self.initial_SE3_init):
                weight_name = "initial_SE3_%05d" % index
                self.initial_SE3.append(   self.add_weight(name=weight_name,shape=(4,4),initializer='identity',trainable=False))
        else:
            weight_name = "initial_SE3"
            self.initial_SE3.append(self.add_weight(name=weight_name,shape=(4,4),initializer='identity',trainable=False))
        super(SE3AccumulationLayer, self).build(input_shape)  # Be sure to call this somewhere!
    def reset_states(self):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        self.initial_SE3 = [K.eye(4) for x in self.initial_SE3]
    def call(self, x):
        if self.stateful:
            output_list = []
            for index in range(self.batch_size):
                current_matrix = x[index]
                prev_matrix = self.initial_SE3[index];
                current_cumulative = tf.matmul(current_matrix,prev_matrix)
                #iself.initial_SE3[index]  = current_cumulative
                output_list.append(current_cumulative)
            if(self.batch_size and self.batch_size > 0):
                output_tensor = K.stack(output_list)
                updates = list(zip(self.initial_SE3,output_list))
                #self.add_update(updates,x)
            else:
                output_tensor = K.stack([K.eye(4)])
            return output_tensor
        else:
            output_list = []
            prev_matrix = self.initial_SE3[0]
            for index in range(self.batch_size):
                current_matrix = x[index]
                current_cumulative = tf.matmul(current_matrix,prev_matrix)
                prev_matrix = current_cumulative
                output_list.append(current_cumulative)
            if(self.batch_size and self.batch_size > 0):
                output_tensor = K.stack(output_list)
            else:
                output_tensor = K.stack([K.eye(4)])
            return output_tensor

    def compute_output_shape(self, input_shape):
        return (input_shape[0],4,4)
    def set_initial_SE3(self,initial_SE3):
        self.initial_SE3 = initial_SE3
        self.set_weights(initial_SE3)

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
    

def getModel(height = 384, width = 512,batch_size=32,use_SE3=True,stateful=True,loss_weights=[1000.0,1.0],num_lstm_units= 1000):
    print "Generating model with height={}, width={},batch_size={},use_SE3={},stateful={},lstm_units={},loss_weights={}".format(height,width,batch_size,use_SE3,stateful,num_lstm_units,loss_weights)

    ## convolution model
    conv_activation = lambda x: activations.relu(x,alpha=0.1) # Use the activation from the FlowNetC Caffe implementation
    # left and model
    input_l = Input(batch_shape=(batch_size,height, width, 3), name='pre_input')
    input_r = Input(batch_shape=(batch_size,height, width, 3), name='nxt_input')
    #layer 1
    conv1 = Convolution2D(64,(7,7), batch_size=batch_size, padding = 'same', name = 'conv1',activation=conv_activation)
    conv1_l = conv1(input_l)
    conv1_r = conv1(input_r)
    conv1_pool_l = MaxPooling2D(name='maxpool1_l')(conv1_l)
    conv1_pool_r = MaxPooling2D(name='maxpool1_r')(conv1_r)
    
    #layer 2
    conv2 = Convolution2D(128, (5, 5), padding = 'same', name='conv2',activation=conv_activation)
    conv2_l = conv2(conv1_pool_l)
    conv2_r = conv2(conv1_pool_r)
    conv2_pool_l = MaxPooling2D(name='maxpool2_l')(conv2_l)
    conv2_pool_r = MaxPooling2D(name='maxpool2_r')(conv2_r)

    #layer 3
    conv3 = Convolution2D(256, (5, 5), padding = 'same', name='conv3',activation=conv_activation)
    conv3_l = conv3(conv2_pool_l)
    conv3_r = conv3(conv2_pool_r)
    conv3_pool_l = MaxPooling2D(name='maxpool3_l')(conv3_l)
    conv3_pool_r = MaxPooling2D(name='maxpool3_r')(conv3_r)


    # merge
    print "Generating Correlation layer..."
    add_layer = get_correlation_layer(conv3_pool_l, conv3_pool_r,max_displacement=20,stride2=2,height_8=height/8,width_8=width/8)

    # merged convolution
    conv3_l_redir = Convolution2D(32,(1,1),name="conv_redir",activation=conv_activation)(conv3_pool_l)
    conv3_l_with_corr = concatenate([conv3_l_redir,add_layer],name="concatenated_correlation")
    conv3_1 = Convolution2D(256, (3, 3), padding = 'same', name='conv3_1',activation=conv_activation)(conv3_l_with_corr)
    conv4 = Convolution2D(512, (3, 3), padding = 'same', name='conv4',activation=conv_activation)(conv3_1)
    conv4 = MaxPooling2D(name='maxpool4')(conv4)
    height_16 = height/16; width_16 = width/16
    conv4_1 = Convolution2D(512, (3, 3), padding = 'same', name='conv4_1',activation=conv_activation)(conv4)
    conv5 = Convolution2D(512, (3, 3), padding = 'same', name='conv5',activation=conv_activation)(conv4_1)
    conv5 = MaxPooling2D(name='maxpool5')(conv5)
    height_32 = height_16/2; width_32 = width_16/2
    conv5_1 = Convolution2D(512, (3, 3), padding = 'same', name='conv5_1',activation=conv_activation)(conv5)
    conv6 = Convolution2D(1024, (3, 3), padding = 'same', name='conv6',activation=conv_activation)(conv5_1)
    conv6 = MaxPooling2D(name='maxpool6')(conv6)
    height_64 = height_32/2; width_64 = width_32/2
    flatten_image = Flatten(name="pre_lstm_flatten")(conv6)

    print "Generating LSTM layer..."
    ## inertial model
    input_imu = Input(batch_shape=(batch_size,10, 6), name='imu_input')
    imu_output_width = 15
    
    imu_lstm_0 = LSTM(imu_output_width, name='imu_lstm_0',stateful=stateful,return_sequences=True)(input_imu)
    imu_lstm_1 = LSTM(imu_output_width, name='imu_lstm_1',stateful=stateful,return_sequences=True)(imu_lstm_0)
    imu_lstm_out = LSTM(imu_output_width, name='imu_lstm_out',stateful=stateful)(imu_lstm_1)

    ## core LSTM
    core_lstm_input = concatenate([flatten_image, imu_lstm_out])
    core_lstm_input_width = 1024*height_64*width_64+imu_output_width
    core_lstm_reshaped = Reshape((1, core_lstm_input_width))(core_lstm_input) # 384 * 512

    if stateful:
        # core_lstm = Reshape((1, 97284))(core_lstm) # 375 * 1242
        core_lstm = LSTM(num_lstm_units, name='core_lstm',stateful=stateful)(core_lstm_reshaped)
        core_lstm_output = Dense(6,name='core_lstm_output')(core_lstm)
    else:
        # Use batch dimension as the time dimension
        core_lstm_unbatched = Lambda(lambda x: tf.transpose(x,[1,0,2]),name="unbatch_permutation")(core_lstm_reshaped)
        core_lstm = LSTM(num_lstm_units, name='core_lstm',stateful=stateful,return_sequences=True)(core_lstm_unbatched)
        core_lstm_rebatched = Lambda(lambda x: tf.transpose(x,[1,0,2]),name="rebatch_permutation")(core_lstm)
        core_lstm_flattened = Flatten(name="lstm_batch_flatten")(core_lstm_rebatched)
        core_lstm_output = Dense(6,name='core_lstm_output')(core_lstm_flattened)
    
    print "Generating se3 to SE3 upgrading layer..."
    # Handle frame-to-frame se3 outputs
    skew_vector = Lambda(batch_se3_to_skew_param,name='skew_param')(core_lstm_output)
    
    position_vector = Lambda(batch_se3_to_delta_position,name='se3_v')(core_lstm_output)
    
    # generate frame-to-frame SE3 outputs
    SO3_matrix = Lambda(batchSkewParamToSO3,name='SO3_matrix')(skew_vector)
    position_matrix = Reshape((3,1),name='position_matrix')(position_vector)
    SE3_delta_3x4 = concatenate([SO3_matrix,position_matrix],axis=2,name='SE3_delta_3x4')
    current_SE3_delta = SE3ExpansionLayer(name='SE3_delta_4x4')(SE3_delta_3x4)
    
    # accumulate SE3_deltas into a current SE3
    SE3 = SE3AccumulationLayer(name='SE3_accumulation',batch_size=batch_size,stateful=stateful)(current_SE3_delta)

    # whole model
    output_list = [skew_vector,position_vector]
    loss_list = ['mean_squared_error','mean_squared_error']
    if(use_SE3):
        output_list += [SE3,SE3]
        loss_list += [loss_angle_SE3,loss_position_SE3]
	if len(loss_weights) ==2:
		loss_weights += loss_weights
    model = Model(inputs = [input_l, input_r, input_imu], outputs = output_list)
    #model = Model(inputs = [input_l, input_r, input_imu], outputs = core_lstm)

    print "Compiling..."
    optimizer = SGD(nesterov=True, lr=0.0000001, momentum=0.0,decay=0.000);
    optimizer = SGD(nesterov=True, lr=0.000001, momentum=0.1,decay=0.001);
    model.compile(optimizer=optimizer,loss=loss_list,loss_weights=loss_weights)
    #model.compile(optimizer=SGD(lr=1, momentum=0.9, nesterov=True),loss=loss_list)
    print "Done"

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
        mat = np.dot(mat_lst[i+1], np.linalg.inv(mat_lst[i]))
        w = logm(mat[:3, :3])
        w = np.array([w[2,1], w[0,2], w[1,0]]).reshape(1,3)
        v = mat[:3, 3].reshape(1,3)
        ans_lst.append(np.concatenate([w, v], axis = 1))
        
    p_img_lst = np.array(p_img_lst)
    n_img_lst = np.array(n_img_lst)
    imu_lst = np.array(imu_lst)
    ans_lst = np.array(ans_lst)
    return p_img_lst, n_img_lst, imu_lst, ans_lst

#path = "../../dataset/mav0/mav0/"
#path = ""


## left image generator
def loadLeftImage(path, batch_size = 32,height = 384, width=512):
    batch_size = 1
    while (True):
        cam0_path = path + "cam0/cam0_align.csv"
        cam0 = pd.read_csv(cam0_path, header = None).drop([0, 1], axis = 1)
        cam0 = list(np.array(cam0).flatten())
        cam0.pop(-1)
        size = len(cam0)
        #size = 3
        if (size % batch_size == 0):
            num_batch = size / batch_size
        else:
            num_batch = size / batch_size + 1
        for i in xrange(num_batch):
            imgs = []
            for j in xrange(i * batch_size, min((i+1) * batch_size, size)):
                img = load_img(path + "cam0/data/" + cam0[j], target_size=(height, width))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                imgs.append(img)
            imgs = np.concatenate(imgs, axis = 0)
            print "left image: " + str(imgs.shape)
            yield imgs

## right image generator
def loadRightImage(path, batch_size = 32,height = 384, width=512):
    batch_size = 1
    while (True):
        cam0_path = path + "cam0/cam0_align.csv"
        cam0 = pd.read_csv(cam0_path, header = None).drop([0, 1], axis = 1)
        cam0 = list(np.array(cam0).flatten())
        cam0.pop(0)
        size = len(cam0)
        #size = 3
        if (size % batch_size == 0):
            num_batch = size / batch_size
        else:
            num_batch = size / batch_size + 1
        for i in xrange(num_batch):
            imgs = []
            for j in xrange(i * batch_size, min((i+1) * batch_size, size)):
                img = load_img(path + "cam0/data/" + cam0[j], target_size=(height, width))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                imgs.append(img)
            imgs = np.concatenate(imgs, axis = 0)
            print "right image: " + str(imgs.shape)
            yield imgs

## imu data generator
def loadImu(path, batch_size = 32):
    batch_size = 1
    while (True):
        imu0_path = path + "imu0/imu0_align.csv"
        imu0 = pd.read_csv(imu0_path, header = None).drop(0, axis = 1)
        imu = np.array(imu0, dtype = np.float64)
        #size = imu.shape[0]
        size = 30
        if (size % (batch_size * 10) == 0):
            num_batch = size / (batch_size * 10)
        else:
            num_batch = size / (batch_size * 10) + 1
        for i in xrange(num_batch):
            data = imu[i * 10 * batch_size: min(size, (i+1) * 10 * batch_size)]
            data = data.reshape((-1, 10, 6))
            print "imu shape:" + str(data.shape)
            yield data

def pqToM(p,q):
    qr = float(q[0])
    qi = float(q[1])
    qj = float(q[2])
    qk = float(q[3])
    M = np.zeros([4,4])
    M[0,0] = 1 - 2*qj*qj - 2*qk*qk
    M[1,0] = 2*(qi*qj+qk*qr)
    M[2,0] = 2*(qi*qk-qj*qr)
    M[3,0] = 0
    M[0,1] = 2*(qi*qj-qk*qr)
    M[1,1] = 1-2*qi*qi-2*qk*qk
    M[2,1] = 2*(qi*qr+qj*qk)
    M[3,1] = 0
    M[0,2] = 2*(qi*qk+qj*qr)
    M[1,2] = 2*(qj*qk-qi*qr)
    M[2,2] = 1-2*(qi*qi+qj*qj)
    M[3,2] = 0
    M[0,3] = float(p[0])
    M[1,3] = float(p[1])
    M[2,3] = float(p[2])
    M[3,3] = 1
    return M

## ground truth generator
def loadGrndTruth(path, batch_size = 32):
    batch_size = 1
    while (True):
        grndTruth_path = path + "state_groundtruth_estimate0/ground_truth_align.csv"
        gndTurth_raw = pd.read_csv(grndTruth_path, header=None).drop(0, axis = 1)
        p1 = np.array(gndTurth_raw[[2,3,4]])
        q1 = np.array(gndTurth_raw[[5,6,7,8]])
        size = p1.shape[0]
        #size = 30
        M_initial = np.linalg.inv(pqToM(p1[0],q1[0]))
        index = [ind for ind in xrange(size) if ind % 10 == 9]
        Ms = []
        if (len(index) % batch_size == 0):
            num_batch = len(index) / batch_size
        else:
            num_batch = len(index) / batch_size + 1
        for i in xrange(num_batch):
            indices = index[i * batch_size: min(len(index), (i + 1) * batch_size)]
            p = p1[indices]
            q = q1[indices]
            ws = []
            vs = []
            M_invs = []
            for idx in xrange(p.shape[0]):
                currentQ = q[idx]
                currentP = p[idx]
                M = pqToM(currentP,currentQ)
                M_inv = np.linalg.inv(M)
                if (len(Ms) == 0):
                    M_last = M_inv
                else:
                    M_last = Ms[-1]
                SE = np.dot(M_inv, np.linalg.inv(M_last))
                v = SE[:3, -1].reshape((1,3))
                wx = SE[:3, :3]
                wx = scipy.linalg.logm(wx)
                w = np.array([-wx[1, 2], wx[0, 2], -wx[0, 1]]).reshape((1,3))
                Ms.append(M_inv)
                M_inv = np.expand_dims(M_inv, axis = 0)
                ws.append(w)
                vs.append(v)
                M_invs.append(M_inv)
            ws = np.concatenate(ws, axis = 0)
            vs = np.concatenate(vs, axis = 0)
            M_invs = np.concatenate(M_invs, axis = 0)
            print "w shape: " + str(ws.shape)
            print "v shape: " + str(vs.shape)
            print "M_inv shape: " + str(M_invs.shape)
            yield [ws, vs, M_invs, M_invs,M_initial]
            M_initial = Ms[-1]

def skewParamToSO3(skew_vector):
    x = skew_vector[0]
    y = skew_vector[1]  
    z = skew_vector[2]
    exponent = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    SO3 = scipy.linalg.expm(exponent)
    return SO3

def skewVToSE3(s,v):
    SO3 = skewParamToSO3(s)
    SE3_delta_3x4 = np.hstack([SO3,np.reshape(v,(3,1))])
    SE3_delta_4x4 = np.vstack([SE3_delta_3x4,np.array([[0,0,0,1]])])
    return SE3_delta_4x4
# def trainingGenerator(num_sequence = 1):
#     lefts = []
#     rights = []
#     imus = []
#     ws = []
#     vs = []
#     M_invs = []
#     M_initials = []
#     for i in xrange(num_sequence):
#         path = 
#         left = loadLeftImage(path = path)
#         right = loadRightImage(path = path)
#         imu = loadImu(path = path)
#         ground = loadGrndTruth(path = path)
#         w = ground[0]
#         v = ground[1]
#         M_inv = ground[2]
#         M_initial = ground[4]
#         ws.append(w)
#         vs.append(v)
#         M_invs.append(M_inv)
#         M_initials.append(M_initial)
#         lefts.append(left)
#         rights.append(right)
#         imus.append(imu)
#     lefts = np.concatenate(lefts, axis = 0)
#     rights = np.concatenate(rights, axis = 0)
#     imus = np.concatenate(imus, axis = 0)
#     ws = np.concatenate(ws, axis = 0)
#     vs = np.concatenate(vs, axis = 0)
#     M_invs = np.concatenate(M_invs, axis = 0)
#     M_initials = np.concatenate(M_initials, axis = 0)
#     grounds = [ws, vs, M_invs, M_invs, M_initials]
#     yield lefts, rights, imus, grounds

# def testGenerator(num_sequence = 1):
#     path = 
#     left = loadLeftImage(path = path)
#     right = loadRightImage(path = path)
#     imu = loadImu(path = path)
#     ground = loadGrndTruth(path = path)
#     w = ground[0]
#     v = ground[1]
#     M_inv = ground[2]
#     M_initial = ground[4]
#     lefts = np.repeat(left, num_sequence, axis = 0)
#     rights = np.repeat(right, num_sequence, axis = 0)
#     imus = np.repeat(imu, num_sequence, axis = 0)
#     ws = np.repeat(w, num_sequence, axis = 0)
#     vs = np.repeat(v, num_sequence, axis = 0)
#     M_invs = np.repeat(M_inv, num_sequence, axis = 0)
#     M_initials = np.repeat(M_initial, num_sequence, axis = 0)
#     grounds = [ws, vs, M_invs, M_invs, M_initials]
#     yield lefts, rights, imus, grounds

## generator for kitti dataset

def groundToMat(line):
    ans = map(float, line.strip().split(' '))
    mat = np.reshape(ans + [0, 0, 0, 1], (4, 4))
    return mat

#left image
def loadKittiLeftImage(path, size, batch_size = 32,height=384,width=512):
    img_folder = path + 'image_00/data/'
    img_list = os.listdir(img_folder)
    while (True):
        if ((size-1) % batch_size == 0):
            num_batch = (size-1) / batch_size
        else:
            num_batch = (size-1) / batch_size + 1
        for i in xrange(num_batch):
            imgs = []
            for j in xrange(i * batch_size, min((i+1) * batch_size, size-1)):
                #img_file = '%s%010d.png' % (img_folder, j)
                img_file = img_folder+img_list[j]
                img = load_img(img_file, target_size=(height, width))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                imgs.append(img)
            imgs = np.concatenate(imgs, axis = 0)
            #print "left image: " + str(imgs.shape)
            yield imgs
            

# right image
def loadKittiRightImage(path, size, batch_size = 32,height=384,width=512):
    img_folder = path + 'image_00/data/'
    img_list = os.listdir(img_folder)
    while (True):
        if ((size-1) % batch_size == 0):
            num_batch = (size-1) / batch_size
        else:
            num_batch = (size-1) / batch_size + 1
        for i in xrange(num_batch):
            imgs = []
            for j in xrange(i * batch_size+1, min((i+1) * batch_size+1, size)):
                #img_file = '%s%010d.png' % (img_folder, j)
                img_file = img_folder+img_list[j]
                img = load_img(img_file, target_size=(height, width))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                imgs.append(img)
            imgs = np.concatenate(imgs, axis = 0)
            #print "left image: " + str(imgs.shape)
            yield imgs

# imu data
def loadKittiImu(path, size, batch_size = 32):
    imu_folder = path + "oxts/data/"
    imu_files = listdir(imu_folder)
    while (True):
        if ((size-1) % batch_size == 0):
            num_batch = (size-1) / batch_size
        else:
            num_batch = (size-1) / batch_size + 1
        for i in xrange(num_batch):
            imus = []
            for j in xrange(i * batch_size*10, min((i+1) * batch_size*10, (size-1)*10)):
                imu_file = imu_folder + imu_files[j]
                with open(imu_file) as f:
                    line = f.readline()
                    tmp = line.strip().split(' ')
                    imu = np.array(tmp[11:14] + tmp[17:20])
                    imu = np.expand_dims(imu, axis=0)
                    imus.append(imu)
            imus = np.array(imus)
            imus = imus.reshape((-1,10,6))
            #print "imu shape: " + str(imus.shape)
            yield imus

# ground truth
def loadKittiGrndTruth(path, size, batch_size = 32):
    ans_file = path + "pose.txt"
    while (True):
        if ((size-1) % batch_size == 0):
            num_batch = (size-1) / batch_size
        else:
            num_batch = (size-1) / batch_size + 1
        Ms = []
        with open(ans_file) as f:
            lines = f.readlines()
            M_initial = np.linalg.inv(groundToMat(lines[0]))
            for i in xrange(num_batch):
                M_invs = []
                ws = []
                vs = []
                for j in xrange(i * batch_size+1, min((i+1) * batch_size+1, size)):
                    line = lines[j]
                    M = groundToMat(line)
                    M_inv = np.linalg.inv(M)
                    if (len(Ms) == 0):
                        M_last = M_initial
                    else:
                        M_last = Ms[-1]
                    SE = np.dot(M_inv, np.linalg.inv(M_last))
                    v = SE[:3, -1].reshape((1,3))
                    wx = SE[:3, :3]
                    wx = scipy.linalg.logm(wx)
                    w = np.array([-wx[1, 2], wx[0, 2], -wx[0, 1]]).reshape((1,3))
                    Ms.append(M_inv)
                    M_inv = np.expand_dims(M_inv, axis = 0)
                    ws.append(w)
                    vs.append(v)
                    M_invs.append(M_inv)
                ws = np.concatenate(ws, axis = 0)
                vs = np.concatenate(vs, axis = 0)
                M_invs = np.concatenate(M_invs, axis = 0)
                #print "w shape: " + str(ws.shape)
                #print "v shape: " + str(vs.shape)
                #print "M_inv shape: " + str(M_invs.shape)
                yield [ws, vs, M_invs, M_invs, M_initial]
                M_initial = Ms[-1]

g_kitti_paths = ["../../dataset/2011_09_30_drive_0020_sync/",
            "../../dataset/2011_09_30_drive_0028_sync/",
            "../../dataset/2011_09_30_drive_0034_sync/",
            "../../dataset/2011_10_03_drive_0034_sync/",
            "../../dataset/2011_09_30_drive_0018_sync/",
            "../../dataset/2011_09_30_drive_0027_sync/",
            "../../dataset/2011_09_30_drive_0033_sync/",
            "../../dataset/2011_10_03_drive_0027_sync/",
            "../../dataset/2011_09_30_drive_0016_sync/"]
def trainingKittiGenerator(num_sequence = 1,height = 384,width=512,path_offset=0,shuffle_sequences=False):
    paths = ["../../dataset/2011_09_30_drive_0020_sync/",
            "../../dataset/2011_09_30_drive_0028_sync/",
            "../../dataset/2011_09_30_drive_0034_sync/",
            "../../dataset/2011_10_03_drive_0034_sync/",
            "../../dataset/2011_09_30_drive_0018_sync/",
            "../../dataset/2011_09_30_drive_0027_sync/",
            "../../dataset/2011_09_30_drive_0033_sync/",
            "../../dataset/2011_10_03_drive_0027_sync/",
            "../../dataset/2011_09_30_drive_0016_sync/"]
    # Create all Generators
    generators = []
    for i in range(num_sequence):
        if shuffle_sequences:
            if path_offset==0:
                random.shuffle(g_kitti_paths)
            path = g_kitti_paths[(i+path_offset) % len(g_kitti_paths)]
        else:
            path = paths[(i+path_offset) % len(paths)]
        img_folder = path + "image_00/data/"
        imu_folder = path + "oxts/data/"
        ans_file = path + "pose.txt"
        num_imgs = len(listdir(img_folder))
        num_imu = len(listdir(imu_folder))
        num_ground_truth = 0
        with open(ans_file) as f:
            myLines = f.readlines()
            num_ground_truth = len(myLines)
        size = min(num_imgs,num_imu/10,num_ground_truth)
        left = iter(loadKittiLeftImage(path = path, size = size, batch_size = 1,height = height,width=width))
        right = iter(loadKittiRightImage(path = path, size = size, batch_size= 1,height = height,width=width))
        imu = iter(loadKittiImu(path = path, size = size, batch_size = 1))
        ground = iter(loadKittiGrndTruth(path = path, size = size, batch_size = 1))
        generators.append([left,right,imu,ground])
    while 1:
        lefts = []
        rights = []
        imus = []
        ws = []
        vs = []
        M_invs = []
        M_initials = []
        
        
        #Loop through generators for each call of yield
        for i in xrange(num_sequence):
            iters = generators[i]
            left = next(iters[0])
            right = next(iters[1])
            imu = next(iters[2])
            ground = next(iters[3])
            w = ground[0]
            v = ground[1]
            M_inv = ground[2]
            M_initial = ground[4]
            ws.append(w)
            vs.append(v)
            M_invs.append(M_inv)
            M_initials.append(np.expand_dims(M_initial,axis=0))
            lefts.append(left)
            rights.append(right)
            imus.append(imu)
        lefts = np.concatenate(lefts, axis = 0)
        rights = np.concatenate(rights, axis = 0)
        imus = np.concatenate(imus, axis = 0)
        ws = np.concatenate(ws, axis = 0)
        vs = np.concatenate(vs, axis = 0)
        M_invs = np.concatenate(M_invs, axis = 0)
        M_initials = np.concatenate(M_initials, axis = 0)
        grounds = [ws, vs, M_invs, M_invs, M_initials]
        print lefts.shape
        yield lefts, rights, imus, grounds

def testKittiGenerator(num_sequence = 1,height = 384,width=512):
    path = "../../dataset/2011_10_03_drive_0042_sync/"
    imu_folder = path + "oxts/data/"
    size = len(listdir(imu_folder))
    left = iter(loadKittiLeftImage(path = path, size = size, batch_size= 1,height = height,width=width))
    right = iter(loadKittiRightImage(path = path, size = size, batch_size = 1,height = height,width=width))
    imu = iter(loadKittiImu(path = path, size = size, batch_size = 1))
    ground = iter(loadKittiGrndTruth(path = path, size = size, batch_size= 1))
    iters = [left,right,imu,ground]
    while 1:
        left = next(iters[0])
        right = next(iters[1])
        imu = next(iters[2])
        ground = next(iters[3])
        w = ground[0]
        v = ground[1]
        M_inv = ground[2]
        M_initial = ground[4]
        lefts = np.repeat(left, num_sequence, axis = 0)
        rights = np.repeat(right, num_sequence, axis = 0)
        imus = np.repeat(imu, num_sequence, axis = 0)
        ws = np.repeat(w, num_sequence, axis = 0)
        vs = np.repeat(v, num_sequence, axis = 0)
        M_invs = np.repeat(M_inv, num_sequence, axis = 0)
        M_initials = np.repeat(np.expand_dims(M_initial,axis=0), num_sequence, axis = 0)
        grounds = [ws, vs, M_invs, M_invs, M_initials]
        print lefts.shape
        yield lefts, rights, imus, grounds

if __name__ == '__main__':
    height = 200
    width = 540
    SE3_advantage = 4;
    a = 1000.0/np.sqrt(SE3_advantage);
    b = 1/np.sqrt(SE3_advantage);
    loss_weights = [a,b,SE3_advantage*a,SE3_advantage*b]
    model = getModel(height=height, width=width,batch_size = batch_size,stateful=stateful,loss_weights=loss_weights,use_SE3 =  use_SE3,num_lstm_units=500)
    #print model.metrics_names
    # model.summary()

    # generator = zip(loadLeftImage(a, b, batch_size = 32), loadRightImage(a, b, batch_size = 32),
    #     loadImu(a, b, batch_size = 32))
    # output_gene = loadGrndTruth(batch_size = 32)
    # #for input_tensor, output_tensor in zip()
    # model.fit_generator(zip(generator, output_gene), steps_per_epoch=114, epochs = 1)

    '''
    # training for kitti dataset
    path = "../test_data2/"
    img_folder = path + "image_00/data/"
    files = [f for f in listdir(img_folder) if isfile(join(img_folder, f)) and f != ".DS_Store"]
    num = len(files)

    for left_image, right_image, imu, target in zip(loadKittiLeftImage(batch_size = batch_size), 
        loadKittiRightImage(batch_size = batch_size), loadKittiImu(batch_size = batch_size), 
        loadKittiGrndTruth(batch_size = batch_size)):
        initial_train_SE3 = [(target[4])]
        model.layers[-1].set_initial_SE3(initial_train_SE3);
        model.train_on_batch(x=[left_image, right_image, imu], y=target[0:num_targets])
        initial_test_SE3 = [(test_target[4])]
        model.layers[-1].set_initial_SE3(initial_test_SE3);
        score = model.test_on_batch(x=[test_left_image, test_right_image, test_imu], y=test_target[0:num_targets])

        print "score: " + str(score)
        result.append(score)

    '''
    train_losses =[]
    test_losses = []
    print_frequency = 10
    train_iters_per_epoch = 2700
    test_iters_per_epoch = 1100
    offsets = range(0,num_train_sets,batch_size)
    num_offsets = len(offsets)
    for i in range(num_epochs):
        for path_offset in offsets:
            train_iter = 0
            model.reset_states()
            for left_image, right_image, imu, target in trainingKittiGenerator(num_sequence = batch_size,height=height,width=width,path_offset=path_offset):
                train_iter +=1
                initial_train_SE3 = [np.copy(initial_pose) for initial_pose in target[4]]
                model.layers[-1].set_initial_SE3(initial_train_SE3);
                current_losses= model.train_on_batch(x=[left_image, right_image, imu], y=target[0:num_targets])
                train_losses.append(current_losses)
                if (train_iter % print_frequency) == 0:
                    print "Epoch {}, train iter {}, sequence_id {}, loss {}".format(i+1,train_iter,path_offset,current_losses)
                if train_iter >= train_iters_per_epoch:
                    break;
        
        test_iter = 0
        model.reset_states()
        for left_image, right_image, imu, target in testKittiGenerator(num_sequence = batch_size,height=height,width=width):
            test_iter+=1
            initial_test_SE3 = [np.copy(initial_pose) for initial_pose in target[4]]
            model.layers[-1].set_initial_SE3(initial_test_SE3);
            current_losses = model.test_on_batch(x=[left_image, right_image, imu], y=target[0:num_targets])
            test_losses.append(current_losses)
            if (test_iter % print_frequency) == 0:
                print "Epoch {}, test  iter {}, loss {}".format(i+1,test_iter,current_losses)
            if test_iter >= test_iters_per_epoch:
                break;          
        model.save('../mdl/model_{}_epochs.h5'.format(i+1))
        model.save_weights('../mdl/model_weights_{}_epochs.hdf5'.format(i+1))
    print "Generating plots..."
    train_losses_np = np.array(train_losses)
    test_losses_np = np.array(test_losses)
    np.savetxt("test_losses.txt",test_losses_np)
    np.savetxt("train_losses.txt",train_losses_np)
    plot_train_x =(np.arange(len(train_losses_np),dtype=np.float64) +1)/(train_iters_per_epoch*num_offsets)
    plot_test_x = (np.arange(len(test_losses_np),dtype=np.float64)+1)/test_iters_per_epoch
    plt.figure(1)
    plt.cla()
    plt.plot(plot_train_x, train_losses_np[:,0],plot_test_x, test_losses_np[:,0])
    plt.legend(["Train losses", "Test losses"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total losses")
    plt.savefig("total_losses.png")

    plt.figure(2)
    plt.cla()
    plt.plot(plot_train_x, train_losses_np[:,1])
    plt.plot(plot_test_x, test_losses_np[:,1])
    plt.plot(plot_train_x, train_losses_np[:,3])
    plt.plot(plot_test_x, test_losses_np[:,3])
    
    plt.legend(["Train loss se3", "Test loss se3","Train loss SE3", "Test loss SE3"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Attitude Losses")
    plt.savefig("attitude_losses.png")

    plt.figure(3)
    plt.cla()
    plt.plot(plot_train_x, train_losses_np[:,2])
    plt.plot(plot_test_x, test_losses_np[:,2])
    plt.plot(plot_train_x, train_losses_np[:,4])
    plt.plot(plot_test_x, test_losses_np[:,4])
    
    plt.legend(["Train loss se3", "Test loss se3","Train loss SE3", "Test loss SE3"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Position Losses")
    plt.savefig("position_losses.png")


    
 #    cam0_path = path + "cam0/cam0_align.csv"
 #    cam0 = pd.read_csv(cam0_path, header = None).drop([0, 1], axis = 1)
 #    cam0 = list(np.array(cam0).flatten())
 #    size = len(cam0)
 #    left = cam0[-2]
 #    right = cam0[-1]

 #    # left image for test
 #    test_left_image = load_img(path + "cam0/data/" + left, target_size=(384, 512))
 #    test_left_image = img_to_array(test_left_image)
 #    test_left_image = np.expand_dims(test_left_image, axis=0)

 #    # right image for test
 #    test_right_image = load_img(path + "cam0/data/" + right, target_size=(384, 512))
 #    test_right_image = img_to_array(test_right_image)
 #    test_right_image = np.expand_dims(test_right_image, axis=0)

 #    # imu data for test
 #    imu0_path = path + "imu0/imu0_align.csv"
 #    imu0 = pd.read_csv(imu0_path, header = None).drop(0, axis = 1)
 #    imu = np.array(imu0, dtype = np.float64)
 #    test_imu = imu[-10:, :]
 #    test_imu = np.expand_dims(test_imu, axis = 0)

 #    # target for test
 #    grndTruth_path = path + "state_groundtruth_estimate0/ground_truth_align.csv"
 #    gndTurth_raw = pd.read_csv(grndTruth_path, header=None).drop(0, axis = 1)
 #    p1 = np.array(gndTurth_raw[[2,3,4]])
 #    q1 = np.array(gndTurth_raw[[5,6,7,8]])
 #    last_index = size * 10 - 21
 #    indices = size * 10 - 11
 #    p = p1[indices]
 #    q = q1[indices]
 #    p_last = p1[last_index]
 #    q_last = q1[last_index]
 #    qr = float(q[0])
 #    qi = float(q[1])
 #    qj = float(q[2])
 #    qk = float(q[3])
 #    M = np.zeros([4,4])
 #    M[0,0] = 1 - 2*qj*qj - 2*qk*qk
 #    M[1,0] = 2*(qi*qj+qk*qr)
 #    M[2,0] = 2*(qi*qk-qj*qr)
 #    M[3,0] = 0
 #    M[0,1] = 2*(qi*qj-qk*qr)
 #    M[1,1] = 1-2*qi*qi-2*qk*qk
 #    M[2,1] = 2*(qi*qr+qj*qk)
 #    M[3,1] = 0
 #    M[0,2] = 2*(qi*qk+qj*qr)
 #    M[1,2] = 2*(qj*qk-qi*qr)
 #    M[2,2] = 1-2*(qi*qi+qj*qj)
 #    M[3,2] = 0
 #    M[0,3] = float(p[0])
 #    M[1,3] = float(p[1])
 #    M[2,3] = float(p[2])
 #    M[3,3] = 1
 #    M_inv = np.linalg.inv(M)

 #    # last M
 #    qr = float(q_last[0])
 #    qi = float(q_last[1])
 #    qj = float(q_last[2])
 #    qk = float(q_last[3])
 #    M(np.arange(len(train_losses_np))/train_iter_last = np.zeros([4,4])
 #    M_last[0,0] = 1 - 2*qj*qj - 2*qk*qk
 #    M_last[1,0] = 2*(qi*qj+qk*qr)
 #    M_last[2,0] = 2*(qi*qk-qj*qr)
 #    M_last[3,0] = 0
 #    M_last[0,1] = 2*(qi*qj-qk*qr)
 #    M_last[1,1] = 1-2*qi*qi-2*qk*qk
 #    M_last[2,1] = 2*(qi*qr+qj*qk)
 #    M_last[3,1] = 0
 #    M_last[0,2] = 2*(qi*qk+qj*qr)
 #    M_last[1,2] = 2*(qj*qk-qi*qr)
 #    M_last[2,2] = 1-2*(qi*qi+qj*qj)
 #    M_last[3,2] = 0
 #    M_last[0,3] = float(p_last[0])
 #    M_last[1,3] = float(p_last[1])
 #    M_last[2,3] = float(p_last[2])
 #    M_last[3,3] = 1; M_last_inv = np.linalg.inv(M_last)

 #    SE = np.dot(M_inv, M_last)
 #    v = SE[:3, -1].reshape((1,3))
 #    wx = SE[:3, :3]
 #    wx = scipy.linalg.logm(wx)
 #    w = np.array([-wx[1, 2], wx[0, 2], -wx[0, 1]]).reshape((1,3))
 #    M_inv = np.expand_dims(M_inv, axis = 0)
 #    M_last_inv = np.expand_dims(M_last_inv, axis = 0)
 #    test_target = [w, v, M_inv, M_inv, M_last_inv]
    
 #    #Reshape for batch_size
 #    test_left_image = np.repeat(test_left_image, batch_size, axis=0)
 #    test_right_image = np.repeat(test_right_image, batch_size, axis=0)
 #    test_imu = np.repeat(test_imu, batch_size, axis=0)
 #    test_target = [np.repeat(i,batch_size,axis=0) for i in test_target]
 #    print "w shape for test: " + str(w.shape)
 #    print "v shape for test: " + str(v.shape)
 #    print "M_inv shape for test: " + str(M_inv.shape)
 #    print "left image for test: " + str(test_left_image.shape)
 #    print "right image for test: " + str(test_right_image.shape)
 #    print "imu for test: " + str(test_imu.shape)

 #    print 'batch_size = %d' % batch_size

 #    result = []
 #    print "Starting training..."
 #    for left_image, right_image, imu, target in itertools.izip(loadLeftImage(batch_size = batch_size), 
 #        loadRightImage(batch_size = batch_size), loadImu(batch_size = batch_size), 
 #        loadGrndTruth(batch_size = batch_size)):
 #        initial_train_SE3 = [(target[4])]
 #        model.layers[-1].set_initial_SE3(initial_train_SE3);
 #        x= [left_image,right_image,imu]
	# y = target[0:num_targets]
	# #print '[train start]'	
 #        model.train_on_batch(x=x, y=y)
	# #print '[train end]'	
 #        initial_test_SE3 = [(test_target[4][0])]
 #        model.layers[-1].set_initial_SE3(initial_test_SE3);
	# #print '[test start]'	
 #        score = model.test_on_batch(x=[test_left_image, test_right_image, test_imu], y=test_target[0:num_targets])
	# # print '[test end]'

 #        print "score: " + str(score)
 #        result.append(score)

 #    model.save('../mdl/model_1e.h5')

 #    print "Starting focused training..."

 #    for index in range(50):
 #        initial_test_SE3 = [(test_target[4][0])]
 #        model.layers[-1].set_initial_SE3(initial_test_SE3);
 #        foo = model.train_on_batch(x=[test_left_image, test_right_image, test_imu], y=test_target[0:num_targets])
 #        initial_test_SE3 = [(test_target[4][0])]
 #        model.layers[-1].set_initial_SE3(initial_test_SE3);
 #        score = model.test_on_batch(x=[test_left_image, test_right_image, test_imu], y=test_target[0:num_targets])

 #        print "score: " + str(score)
 #        result.append(score)

    # plt.figure()
    # plt.plot(range(len(result)), result)
    # plt.savefig("result.png")



    #p_img_lst, n_img_lst, imu_lst, ans_lst = readData()
    #model.fit({'pre_input': p_img_lst, 'nxt_input': n_img_lst, 'imu_input': imu_lst}, \
    #        {'output': ans_lst}, epochs=1, batch_size=1)
