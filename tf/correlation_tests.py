#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.

.. moduleauthor:: David Stutz
"""

import unittest
import numpy as np
import tensorflow as tf
import _correlation_grad
correlation_module = tf.load_op_library('build/libcorrelation.so')

class CorrelationOpTest(unittest.TestCase):
    def test_raisesExceptionWithIncompatibleDimensions(self):
        with tf.Session(''):
            with self.assertRaises(ValueError):
                correlation_module.correlation([1, 2], [[1, 2], [3, 4]]).eval()
            with self.assertRaises(ValueError):
                self.assertRaises(correlation_module.correlation([1, 2], [1, 2, 3, 4]).eval(), ValueError)
            with self.assertRaises(ValueError):
                self.assertRaises(correlation_module.correlation([1, 2, 3], [[1, 2], [3, 4]]).eval(), ValueError)
            
    def test_innerProductHardCoded(self):
        with tf.Session(''):
            result = correlation_module.correlation([[1], [2]], [[1, 2], [3, 4]]).eval()
            self.assertEqual(result.shape[0], 2)
            self.assertEqual(result[0], 5)
            self.assertEqual(result[1], 11)
    
    def test_innerProductGradientXHardCoded(self):
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape = (2))
            W = tf.constant(np.asarray([[1, 2], [3, 4]]).astype(np.float32))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_correlation = correlation_module.correlation(tf.reshape(x, [-1, 1]), W)
            
            grad_x_tf = tf.gradients(Wx_tf, x)
            grad_x_correlation = tf.gradients(Wx_correlation, x)
            
            gradient_tf = sess.run(grad_x_tf, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
            gradient_correlation = sess.run(grad_x_correlation, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
            
            self.assertEqual(gradient_tf[0][0], gradient_correlation[0][0])
            self.assertEqual(gradient_tf[0][1], gradient_correlation[0][1])
    
    def test_innerProductGradientWHardCoded(self):
        with tf.Session('') as sess:
            x = tf.constant(np.asarray([1, 2]).astype(np.float32))
            W = tf.placeholder(tf.float32, shape = (2, 2))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_correlation = correlation_module.correlation(tf.reshape(x, [-1, 1]), W)
            
            grad_W_tf = tf.gradients(Wx_tf, W)
            grad_W_correlation = tf.gradients(Wx_correlation, W)
            
            gradient_tf = sess.run(grad_W_tf, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float32)})
            gradient_correlation = sess.run(grad_W_correlation, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float32)})
            
            self.assertEqual(gradient_tf[0][0][0], gradient_correlation[0][0][0])
            self.assertEqual(gradient_tf[0][0][1], gradient_correlation[0][0][1])
            self.assertEqual(gradient_tf[0][1][0], gradient_correlation[0][1][0])
            self.assertEqual(gradient_tf[0][1][1], gradient_correlation[0][1][1])
    
    def test_innerProductRandom(self):
        with tf.Session(''):
            n = 4
            m = 5
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n, 1))
                W_rand = np.random.randint(10, size = (m, n))
                result_rand = np.dot(W_rand, x_rand)
                
                result = correlation_module.correlation(x_rand, W_rand).eval()
                np.testing.assert_array_equal(result, result_rand)
    
    def test_innerProductGradientXRandom(self):
        with tf.Session('') as sess:
            n = 4
            m = 5
            
            x = tf.placeholder(tf.float32, shape = (n))
            W = tf.placeholder(tf.float32, shape = (m, n))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_correlation = correlation_module.correlation(tf.reshape(x, [-1, 1]), W)
            
            grad_x_tf = tf.gradients(Wx_tf, x)
            grad_x_correlation = tf.gradients(Wx_correlation, x)
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n))
                W_rand = np.random.randint(10, size = (m, n))
                
                gradient_tf = sess.run(grad_x_tf, feed_dict = {x: x_rand, W: W_rand})
                gradient_correlation = sess.run(grad_x_correlation, feed_dict = {x: x_rand, W: W_rand})
                
                np.testing.assert_array_equal(gradient_tf, gradient_correlation)
                
    def test_innerProductGradientWRandom(self):
        with tf.Session('') as sess:
            n = 4
            m = 5
            
            x = tf.placeholder(tf.float32, shape = (n))
            W = tf.placeholder(tf.float32, shape = (m, n))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_correlation = correlation_module.correlation(tf.reshape(x, [-1, 1]), W)
            
            grad_W_tf = tf.gradients(Wx_tf, W)
            grad_W_correlation = tf.gradients(Wx_correlation, W)
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n))
                W_rand = np.random.randint(10, size = (m, n))
                
                gradient_tf = sess.run(grad_W_tf, feed_dict = {x: x_rand, W: W_rand})
                gradient_correlation = sess.run(grad_W_correlation, feed_dict = {x: x_rand, W: W_rand})
                
                np.testing.assert_array_equal(gradient_tf, gradient_correlation)
                  
                
if __name__ == '__main__':
    unittest.main()