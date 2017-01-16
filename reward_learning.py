import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

# Input is a 28x28 image (flattened to a 1x784 vector)
# Output is a 28x28 reward map (flattened to a 1x784 vector)

state = tf.placeholder(tf.float32, shape=[None, 784])
sentence_vec = tf.placeholder(tf.float32, shape=[None, 4])

# Maybe change to one-hot vector for each location on the grid.
rewards_actual = tf.placeholder(tf.float32, shape=[None, 784, 2])

'''
Data reading function.
'''
def read_data(file_name):

    grids = []
    sentences = []
    rewards = []

    with open(file_name, 'rb') as fp:
        file = pickle.load(fp)
    for grid, sentence, reward in file:
        grids.append(grid)
        sentences.append(sentence)
        rewards.append(reward)

    return np.array(grids), np.array(sentences), np.array(rewards)

def next_batch(data, number, batch_size):
    returned_data = []

    num_rows = data.shape[0]

    start = number*batch_size % num_rows
    end = start + batch_size

    returned_data = data[start:np.min(num_rows, end)]

    if end > num_rows:
        end = end - num_rows
        data2 = data[0:end]
        returned_data = np.concatenate((returned_data, data2), axis=0)

    return returned_data


'''
Network definition functions.
'''
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

# First convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

state_image = tf.reshape(state, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(state_image, W_conv1) + b_conv1)
h_pool1 = h_conv1  # max_pool_2x2(h_conv1) #No max pooling for now

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = h_conv2  # max_pool_2x2(h_conv2) #No max pooling for now

# Fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 50])
b_fc1 = bias_variable([50])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Fully connected layer for the language vector
W_fc2 = weight_variable([4, 50])
b_fc2 = bias_variable([50])
h_fc2 = tf.nn.relu(tf.matmul(sentence_vec, W_fc2) + b_fc2)

# Concatenate fc layers for image and language inputs -> should be a 1x200
# vector
# Is 1 the correct dimension to concatenate along?
h_fc_concatenated = tf.concat(1, [h_fc1, h_fc2])

# Dropout according to input probability
keep_prob = tf.placeholder(tf.float32)
h_fc_concatendated = tf.nn.dropout(h_fc_concatenated, keep_prob)

h_fc_concatenated = tf.reshape(h_fc_concatenated, [-1, 10, 10, 1])

# How to properly do a deconvolution in tensorflow? Are the two lines
# below correct?
W_deconv1 = weight_variable([5, 5, 1, 2])
h_deconv1 = tf.nn.conv2d_transpose(h_fc_concatenated, filter=W_deconv1, output_shape=[
                                   1, 28, 28, 2], strides=[1, 1, 1, 1], padding='SAME')

# Should I include another step between the deconvolution and outputting
# the predicted reward map?

# These are the predicted rewards
rewards_pred = tf.reshape(h_deconv1, [-1, 784, 2])



print rewards_pred.get_shape()

loss_array = np.array([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(rewards_pred[0][i], rewards_actual[0][i])) for i in xrange(784)])

cross_entropy = np.sum(loss_array)

#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(rewards_pred, rewards_actual)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = [tf.equal(
    tf.argmax(rewards_pred[i][j], 1), tf.argmax(rewards_actual[i][j], 1)) for i in xrange(28) for j in xrange(28)]

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())


train_grids, train_sentences, train_rewards = read_data('train')
test_grids, test_sentences, test_rewards = read_data('test')

batch_size = 50

for i in range(2000):
    grids_batch = next_batch(train_grids, i, batch_size)
    sentences_batch = next_batch(train_sentences, i, batch_size)
    rewards_batch = next_batch(train_rewards, i, batch_size)

    train_step.run(feed_dict={state: grids_batch, sentence_vec: sentences_batch, rewards_actual: rewards_batch, keep_prob: 1})

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    

#print("test accuracy: %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images[0], y_: mnist.test.labels[0], keep_prob: 1.0}))
