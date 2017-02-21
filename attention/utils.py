import tensorflow as tf

'''
Network definition functions.
'''
def weight_variable(shape):
	num_nodes = 1

	for i in xrange(len(shape)):
		num_nodes *= shape[i]


    #initial = tf.truncated_normal(shape, stddev=0.1)
	initial = tf.truncated_normal(shape, stddev=tf.sqrt(1/float(num_nodes)))

    #initial = tf.uniform_unit_scaling_initializer(shape, dtype=tf.float32)

	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)

    #initial = tf.constant(0.001, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')