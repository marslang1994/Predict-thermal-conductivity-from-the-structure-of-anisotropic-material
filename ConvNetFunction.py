from __future__ import division
import numpy as np
import tensorflow as tf



#weights initialization.
def init_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

#bias initialization.
def init_bias(size):
	return tf.Variable(tf.constant(0.05, shape = [size]))


# create a convolutional layer with maxpooling.
def conv_layer(input, 
			   channels, 
			   kernel_size, 
			   num_filters):
	weights = init_weights(shape = [kernel_size, kernel_size, channels, num_filters])
	bias = init_bias(num_filters)
	layer = tf.nn.conv2d(input = input,
						 filter = weights,
						 strides = [1, 1, 1, 1],
						 padding = 'SAME')
	layer += bias
	
	layer = tf.nn.max_pool(value = layer,
						   ksize = [1, 2, 2, 1],
						   strides = [1, 2, 2, 1],
						   padding = 'VALID')
	layer = tf.nn.relu(layer)
	return layer

# create a flatten layer.
def flatten(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer = tf.reshape(layer, [-1, num_features])
	return layer, num_features


# create a fully connected layer
def fc_layer(input,
			 num_inputs,
			 num_outputs,
			 use_activiation = True):
	weights = init_weights(shape = [num_inputs, num_outputs])
	bias = init_bias(size = num_outputs)
	layer = tf.matmul(input, weights) + bias
	if use_activiation == True:
		layer = tf.nn.relu(layer)
	return layer



























