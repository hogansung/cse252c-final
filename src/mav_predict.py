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


EPOCH_SIZE = 5
BATCH_SIZE = 1
NUM_INST = 1800

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
    y_true_inverse = tf.matrix_inverse(y_true)
    small_world_errors = tf.matmul(y_true_inverse,y_pred)
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
                self.add_update(updates,x)
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
    
def getModel(height=384, width=512, batch_size=32, use_SE3=True, stateful=True, loss_weights=[1.0,1.0], num_lstm_units=1000):
    print "Generating model with height={}, width={},batch_size={},use_SE3={},stateful={},lstm_units={},loss_weights={}".format(height,width,batch_size,use_SE3,stateful,num_lstm_units,loss_weights)

    ## convolution model
    conv_activation = lambda x: activations.relu(x,alpha=0.1) # Use the activation from the FlowNetC Caffe implementation

    # left and right model
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

    print "Compiling..."
    optimizer = SGD(nesterov=True, lr=0.000001, momentum=0.1,decay=0.001);
    model.compile(optimizer=optimizer,loss=loss_list,loss_weights=loss_weights)
    print "Done"

    return model


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

def data_generator(batch_size=32, target_size=(384,512), num_inst=1800, st_pos=0):
    num_batch = num_inst / batch_size

    # image
    cam0_path = '../../dataset/mav0/mav0/cam0/'
    cam0 = pd.read_csv(cam0_path + 'cam0_align.csv', header = None).drop([0, 1], axis=1)
    cam0 = list(np.array(cam0).flatten())

    # imu
    imu0_path = '../../dataset/mav0/mav0/imu0/'
    imu0 = pd.read_csv(imu0_path + 'imu0_align.csv', header = None).drop(0, axis=1)
    imuLst = np.array(imu0, dtype = np.float64)

    # ground truth
    grnd_path = '../../dataset/mav0/mav0/state_groundtruth_estimate0/'
    grnd = pd.read_csv(grnd_path + 'ground_truth_align.csv', header=None).drop(0, axis=1)
    pLst = np.array(grnd[[2,3,4]])
    qLst = np.array(grnd[[5,6,7,8]])
    M_init = np.linalg.inv(pqToM(pLst[0],qLst[0]))
    M_last = M_init

    for i in xrange(num_batch):
        # img
        imgs = []
        for j in xrange(st_pos + i*batch_size, st_pos + (i+1)*batch_size + 1):
            img = load_img(cam0_path + 'data/' + cam0[j], target_size=target_size)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            imgs.append(img)

        imgs = np.concatenate(imgs, axis = 0)

        # imu
        imu = imuLst[i*10*batch_size: (i+1)*10*batch_size]
        imu = imu.reshape((-1, 10, 6))

        # ground truth
        ws = []
        vs = []
        M_invs = []
        for j in xrange(st_pos + i*batch_size, st_pos + (i+1)*batch_size):
            p = pLst[10*j + 9]
            q = qLst[10*j + 9]
            M_inv = np.linalg.inv(pqToM(p, q))

            SE = np.dot(M_inv, np.linalg.inv(M_last)) # M_inv / M_last
            v = SE[:3, -1].reshape((1,3))
            wx = scipy.linalg.logm(SE[:3, :3])
            w = np.array([-wx[1, 2], wx[0, 2], -wx[0, 1]]).reshape((1,3))

            M_last = M_inv
            M_inv = np.expand_dims(M_inv, axis = 0)
            ws.append(w)
            vs.append(v)
            M_invs.append(M_inv)

        ws = np.concatenate(ws, axis = 0)
        vs = np.concatenate(vs, axis = 0)
        M_invs = np.concatenate(M_invs, axis = 0)

        yield [imgs[:-1], imgs[1:], imu, [ws, vs, M_invs, M_invs, M_init]]
        M_init = M_invs[-1]


if __name__ == '__main__':
    SE3_advantage = 4;
    a = 1000.0/np.sqrt(SE3_advantage);
    b = 1/np.sqrt(SE3_advantage);
    loss_weights = [a,b,SE3_advantage*a,SE3_advantage*b]

    model = getModel(height=384, width=512, batch_size=BATCH_SIZE, stateful=stateful, loss_weights=loss_weights, \
                     use_SE3=use_SE3, num_lstm_units=1000)
    model.load_weights("../mdl/mav_4.h5")
    print 'load model complete'

    # predict on train
    print 'Starting predicting on training set...'
    first = True
    gLst = []
    vLst = []
    model.reset_states()
    for left_image, right_image, imu, target in data_generator(batch_size=1, st_pos=0, num_inst=NUM_INST):
        x = [left_image,right_image,imu]
        y = target[:num_targets]

        if first:  
            first = False
            initial_SE3 = [target[4]]
        
        # predict
        model.layers[-1].set_initial_SE3(initial_SE3);
        p = model.predict_on_batch(x=x)

        # calculate g and v
        all_predict_positions = p[-1]
        all_true_positions = y[-1]
        g = [np.linalg.inv(yy)[:3, 3] for yy in all_true_positions]
        v = [np.linalg.inv(pp)[:3, 3] for pp in all_predict_positions]

        # append g and v
        gLst += g
        vLst += v

        # set initial state for next step
        last_position = all_predict_positions[-1]
        initial_SE3 = [last_position]

    print 'Finish predicting on training set...'

    # write g and v in file
    with open('../res/mav_tn_g_3D.txt', 'w') as wf:
        for g in gLst:
            wf.write(','.join(map(str, g)) + '\n')

    with open('../res/mav_tn_v_3D.txt', 'w') as wf:
        for v in vLst:
            wf.write(','.join(map(str, v)) + '\n')


    # predict on test
    print 'Starting predicting on testing set...'
    first = True
    gLst = []
    vLst = []
    model.reset_states()
    for left_image, right_image, imu, target in data_generator(batch_size=1, st_pos=1800, num_inst=NUM_INST):
        x = [left_image,right_image,imu]
        y = target[:num_targets]

        if first:  
            first = False
            initial_SE3 = [target[4]]
        
        # predict
        model.layers[-1].set_initial_SE3(initial_SE3);
        p = model.predict_on_batch(x=x)

        # calculate g and v
        all_predict_positions = p[-1]
        all_true_positions = y[-1]
        g = [np.linalg.inv(yy)[:3, 3] for yy in all_true_positions]
        v = [np.linalg.inv(pp)[:3, 3] for pp in all_predict_positions]

        # append g and v
        gLst += g
        vLst += v

        # set initial state for next step
        last_position = all_predict_positions[-1]
        initial_SE3 = [last_position]

    print 'Finish predicting on testing set...'

    # write g and v in file
    with open('../res/mav_tt_g_3D.txt', 'w') as wf:
        for g in gLst:
            wf.write(','.join(map(str, g)) + '\n')

    with open('../res/mav_tt_v_3D.txt', 'w') as wf:
        for v in vLst:
            wf.write(','.join(map(str, v)) + '\n')
