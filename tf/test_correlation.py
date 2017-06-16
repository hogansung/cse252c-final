from __future__ import print_function
import numpy as np
import tensorflow as tf
from time import time

NUM_VALUES = 10

corr = tf.load_op_library('./build/libcorrelation.so')

graph = tf.Graph()
myShape = (1,21,21,3)
with graph.as_default():
  input = tf.placeholder(tf.float32, shape = myShape)
  b = tf.placeholder(tf.float32, shape = myShape)
  result = corr.correlation(input, b)

with tf.Session(graph = graph, config = tf.ConfigProto(log_device_placement = True)) as session:
  feed_dict = {
    input: np.ones(myShape),
    b: np.ones(myShape),
  }
  start = time(); print(session.run([result], feed_dict = feed_dict)); end = time(); print(end - start)
