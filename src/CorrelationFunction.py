import tensorflow as tf
correlation_module = tf.load_op_library("../tf/build/libcorrelation.so")

import _correlation_grad

corr = correlation_module.correlation
