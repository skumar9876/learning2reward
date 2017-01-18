'''
Toy example to get familiar with deconvolution in tensorflow.
'''

import tensorflow as tf

sess = tf.Session()
batch_size = 3
output_shape = [batch_size, 28, 28, 2]
strides = [1, 2, 2, 1]

l = tf.constant(0.1, shape=[batch_size, 10, 10, 1])
w = tf.constant(0.1, shape=[3, 3, 2, 1])

#h1 = tf.nn.conv2d_transpose(l, w, output_shape=output_shape, strides=strides, padding='SAME')

output = tf.constant(0.1, shape=output_shape)
expected_l = tf.nn.conv2d(output, w, strides=strides, padding='SAME')
print expected_l.get_shape()


a = tf.constant(0.1, shape=[2,2])
b = tf.constant(0.1, shape=[2,2])
isequal = tf.equal(a, b)
isequal_float = tf.cast(isequal, tf.float32)

print sess.run(isequal)
print sess.run(isequal_float)
#print sess.run(h1)