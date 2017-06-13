import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.engine.topology import Layer
from keras.models import Model, Sequential, load_model
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
from flownet_resume_6 import *

VERSION = str(27) + '_fix10'

if __name__ == '__main__':
    SE3_advantage = 1.0;
    a = 100.0;
    b = 1.0;
    a1 = a/(1+SE3_advantage);
    a2 = SE3_advantage*a1;
    b1 = b/(1+SE3_advantage)
    b2 = SE3_advantage*b1;
    loss_weights = [a1,b1,a2,b2]

    height = 200
    width = 540

    model = getModel(height=height, width=width,batch_size = batch_size,stateful=stateful,loss_weights=loss_weights,use_SE3 =  use_SE3,num_lstm_units=500)
    model.load_weights('../mdl/model_elu_weights_' + str(27) + '_epochs.hdf5')
    print 'load model complete'

    '''
    gLst = []
    vLst = []
    train_iters_per_epoch = 2700
    train_iter = 0
    model.reset_states()
    for left_image, right_image, imu, target in trainingKittiGenerator(num_sequence=batch_size, height=height, \
								       width=width, path_offset=0):
        y = target[:num_targets]
	train_iter += 1

        if train_iter == 1:
            initial_SE3 = [np.copy(initial_pose) for initial_pose in target[4]]

        # predict
        model.layers[-1].set_initial_SE3(initial_SE3);
        p = model.predict_on_batch(x=[left_image, right_image, imu])

        # calculate g and v
        all_true_positions = y[-1]
        all_predict_positions = p[-1]
        g = [np.linalg.inv(yy)[:3, 3] for yy in all_true_positions]
        v = [np.linalg.inv(pp)[:3, 3] for pp in all_predict_positions]

        # append g and v
        gLst += g[:1]
        vLst += v[:1]

        # set initial state for next step
        last_position = all_predict_positions
        initial_SE3 = [np.copy(initial_pose) for initial_pose in last_position]

	if train_iter >= train_iters_per_epoch:
	    break;

    # write g and v in file
    with open('../res/kitti_tn_' + str(VERSION) + '_g_3D.txt', 'w') as wf:
        for g in gLst:
            wf.write(','.join(map(str, g)) + '\n')

    with open('../res/kitti_tn_' + str(VERSION) + '_v_3D.txt', 'w') as wf:
        for v in vLst:
            wf.write(','.join(map(str, v)) + '\n')
    '''

    gLst = []
    vLst = []
    test_iters_per_epoch = 110
    test_iter = 0
    model.reset_states()
    for left_image, right_image, imu, target in testKittiGenerator(num_sequence = batch_size,height=height,width=width):
        y = target[:num_targets]
        test_iter += 1

        if test_iter < 10:
            initial_SE3 = [np.copy(initial_pose) for initial_pose in target[4]]
	print test_iter

        # predict
        model.layers[-1].set_initial_SE3(initial_SE3)
        p = model.predict_on_batch(x=[left_image, right_image, imu])

	# test
	'''
        model.layers[-1].set_initial_SE3(initial_SE3)
	current_losses = model.test_on_batch(x=[left_image, right_image, imu], y=y)
	print current_losses
	'''

        # calculate g and v
        all_true_positions = y[-1]
        all_predict_positions = p[-1]
        g = [np.linalg.inv(yy)[:3, 3] for yy in all_true_positions]
        v = [np.linalg.inv(pp)[:3, 3] for pp in all_predict_positions]

        # append g and v
        gLst += g
        vLst += v

        # set initial state for next step
        last_position = all_predict_positions
        initial_SE3 = [np.copy(initial_pose) for initial_pose in last_position]

        if test_iter >= test_iters_per_epoch:
            break;          
 
    # write g and v in file
    with open('../res/kitti_tt_' + str(VERSION) + '_g_3D.txt', 'w') as wf:
        for g in gLst:
            wf.write(','.join(map(str, g)) + '\n')

    with open('../res/kitti_tt_' + str(VERSION) + '_v_3D.txt', 'w') as wf:
        for v in vLst:
            wf.write(','.join(map(str, v)) + '\n')
