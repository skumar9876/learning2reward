import tensorflow as tf 
sess = tf.InteractiveSession()

#Input is a 28x28 image
#Output is a 28x28 reward map

state = tf.placeholder(tf.float32, shape=[None, 784])
sentence_vec = tf.placeholder(tf.float32, shape=[None, 10])

rewards_actual = tf.placeholder(tf.float32, shape=[None, 784])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
		strides=[1, 2, 2, 1], padding='SAME')

#First convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

state_image = tf.reshape(state_image, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool2, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 100])
b_fc1 = bias_variable([100])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) +  b_fc1)

#Fully connected layer for the language vector
W_fc2 = weight_variable([10, 100])
b_fc2 = bias_variable([100])
h_fc2 = tf.nn.relu(tf.matmul(sentence_vec, 100) + b_fc2)

#Concatenate fc layers for image and language inputs -> should be a 1x200 vector
h_fc_concatenated = tf.concat(1, [h_fc1, h_fc2]) #Is 1 the correct dimension?

keep_prob = tf.placeholder(tf.float32)
h_fc_concatendated = tf.nn.dropout(h_fc_concatenated, keep_prob)

#How to properly do a deconvolution in tensorflow? Are the two lines below correct?
W_deconv1 = weight_variable([5, 1, 1, 1])
h_deconv1 = tf.nn.conv2d_transpose(h_fc_concatenated, filter=W_deconv1, output_shape=[1,28,28,1], strides=[1,1,1,1], padding='SAME')

rewards_pred = tf.reshape(h_deconv1, [-1, 784]) #These are the predicted rewards