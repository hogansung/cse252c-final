import flownet
import numpy as np
import time
import tensorflow as tf
import keras.backend as K


if __name__=='__main__':
    batch_size=3

    height = 200; width = 540;
    loss_weights = [1000.0, 1.0, 0.0, 0.0]
    start = time.time()
    model = flownet.getModel(height=height,width=width,batch_size=batch_size,stateful=False,loss_weights=loss_weights)
    end = time.time()
    model_creation_time = end-start
    
    print "Time for model creation: {}. With batch size {}, height {}, width {}".format(model_creation_time, batch_size,height,width)
    
    left_image = np.random.standard_normal((batch_size,height,width,3))
    right_image = np.random.standard_normal((batch_size,height,width,3))
    imu = np.random.standard_normal((batch_size,10,6))
    ws = np.random.standard_normal((batch_size,3))
    vs = np.random.standard_normal((batch_size,3))
    M_inv = np.random.standard_normal((batch_size,4,4))

    for i in range(5):
      start = time.time()
      foo = model.train_on_batch(x = [left_image,right_image,imu], y = [ws, vs, M_inv, M_inv])
      end = time.time()
      model_train_time = end- start
      print foo
      print "Time for training: {}. With batch size {}".format(model_train_time, batch_size)
    
    for i in range(10):
      start = time.time()
      foo = model.predict_on_batch(x = [left_image,right_image,imu])
      end = time.time()

      model_predict_time = end- start
      print "Time for prediction: {}. With batch size {}, height {}, width {}".format(model_predict_time, batch_size,height,width)
    positions = foo[-1]
    delta_poses_from_accumulator = []
    for i in range(len(positions)-1):                            
        c = positions[i]                                       
        nextC = positions[i+1]
        delta = np.dot(nextC , np.linalg.inv(c))
        delta_poses_from_accumulator.append(c)

