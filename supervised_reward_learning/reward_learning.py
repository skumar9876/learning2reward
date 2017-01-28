import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from random import randint
from matplotlib import cm

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

'''
Returns the next batch of data. 
'''
def next_batch(data, number, batch_size):
    returned_data = []

    num_rows = data.shape[0]

    start = number*batch_size % num_rows
    end = start + batch_size

    returned_data = data[start:np.min((num_rows, end))]

    if end > num_rows:
        end = end - num_rows
        data2 = data[0:end]
        returned_data = np.concatenate((returned_data, data2), axis=0)

    return returned_data

'''
Visualization of results.
'''
def visualize_results(reward_actual, reward_pred, i, save_string):
    reward_actual = np.array(reward_actual)
    reward_actual = reward_actual.reshape((28, 28))

    reward_pred = np.array(reward_pred)
    reward_pred = reward_pred.reshape((28,28))


    mask = np.ones((28,28))
    reward_actual = mask - reward_actual
    reward_pred = mask - reward_pred


    im1 = Image.fromarray(np.uint8(cm.gist_earth(reward_actual)*255))
    im1 = im1.resize((200,200), Image.ANTIALIAS)
    im1.save(save_string + "_actual_" + str(i), 'JPEG')
    
    im2 = Image.fromarray(np.uint8(cm.gist_earth(reward_pred)*255))
    im2 = im2.resize((200,200), Image.ANTIALIAS)
    im2.save(save_string +"_pred_" + str(i), 'JPEG')


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

'''
Network architecture.
'''

sess = tf.InteractiveSession()

# Input is a 28x28 image (flattened to a 1x784 vector)
# Output is a 28x28 reward map (flattened to a 1x784 vector)

state = tf.placeholder(tf.float32, shape=[None, 784])
sentence_vec = tf.placeholder(tf.float32, shape=[None, 4])

# Vector of reward class for each location on the grid.
rewards_actual = tf.placeholder(tf.int64, shape=[None, 784])

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
W_fc1 = weight_variable([28 * 28 * 64, 392])
b_fc1 = bias_variable([392])
h_pool2_flat = tf.reshape(h_pool2, [-1, 28 * 28 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Fully connected layer for the language vector
W_fc2 = weight_variable([4, 392])
b_fc2 = bias_variable([392])
h_fc2 = tf.nn.relu(tf.matmul(sentence_vec, W_fc2) + b_fc2)

# Concatenate fc layers for image and language inputs -> should be a 1x200
# vector
h_fc_concatenated = tf.concat(1, [h_fc1, h_fc2])
#print h_fc_concatenated.get_shape()

# Dropout according to input probability
keep_prob = tf.placeholder(tf.float32)
h_fc_concatendated = tf.nn.dropout(h_fc_concatenated, keep_prob)

# Reshaped 1x784 vector into 28x28 so that I can do deconvolution on this 2d image -> is this ok?
h_fc_concatenated = tf.reshape(h_fc_concatenated, [-1, 28, 28, 1])

size_of_batch = tf.shape(h_fc_concatenated)[0]

# Deconvolution step
W_deconv1 = weight_variable([5, 5, 2, 1])
output_shape = [size_of_batch, 28, 28, 2]
h_deconv1 = tf.nn.conv2d_transpose(h_fc_concatenated, filter=W_deconv1, output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME')

# Should I include another step between the deconvolution and outputting
# the predicted reward map?

# These are the predicted rewards
rewards_pred = tf.reshape(h_deconv1, [-1, 784, 2])

#print rewards_pred.get_shape()

# Old approach of computing cross entropy loss of each pixel and summing: No longer using this.
# loss_array = np.array([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(rewards_pred[0][i], rewards_actual[0][i])) for i in xrange(784)])
# cross_entropy = np.sum(loss_array)

# IMPORTANT:
# rewards_pred has shape batch_size x 784 x 2 since each entry is a vector of length 2 with the probabilities for the two classes of reward.
# Rewards actual has shape batch_sizex784 instead of batch_size x 784 x 2
# Actual reward map contains the indices of the correct reward for each row of the predicted reward output of size batch_sizex784x2
# E.g. (1,0) --> reward class 0, (0,1) --> reward class 1
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rewards_pred, labels=rewards_actual)

# Weight the cross entropy loss more heavily in locations of goal reward (x392)
# COMMENTED OUT
# loss_weighting = tf.multiply(tf.cast(rewards_actual, tf.float32), cross_entropy)
# loss_weighting = tf.scalar_mul(391, loss_weighting)
# cross_entropy = tf.add(loss_weighting, cross_entropy)


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# HERE: Having trouble measuring accuracy correctly.
#correct_prediction = [tf.equal(
#    tf.argmax(rewards_pred[i][j], 1), rewards_actual[i][j]) for i in xrange(size_of_batch) for j in xrange(784)]
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

rewards_pred_flat = tf.argmax(rewards_pred, 2)

rewards_pred_flat_bool = tf.cast(rewards_pred_flat, tf.int64)
rewards_actual_bool = tf.cast(rewards_actual, tf.int64)

false_positives, fp_update_op = tf.contrib.metrics.streaming_false_positives(rewards_pred_flat_bool, rewards_actual_bool, weights=None, 
    metrics_collections=None, updates_collections=None, name=None)

false_negatives, fn_update_op = tf.contrib.metrics.streaming_false_negatives(rewards_pred_flat_bool, rewards_actual_bool, weights=None, 
    metrics_collections=None, updates_collections=None, name=None)

true_positives = tf.reduce_sum(rewards_actual_bool)

sess.run(tf.local_variables_initializer())


correct_prediction = tf.equal(rewards_pred_flat, rewards_actual)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.global_variables_initializer())



'''
Training and Testing loops.
'''

train_grids, train_sentences, train_rewards = read_data('train')
test_grids, test_sentences, test_rewards = read_data('test')

# Batch size used in training.
batch_size = 50

steps = []
accuracy_arr = []
for i in range(1000):
    grids_batch = next_batch(train_grids, i, batch_size)
    sentences_batch = next_batch(train_sentences, i, batch_size)
    rewards_batch = next_batch(train_rewards, i, batch_size)

    #print np.shape(grids_batch)
    #print np.shape(sentences_batch)

    train_step.run(feed_dict={state: grids_batch, sentence_vec: sentences_batch, rewards_actual: rewards_batch, keep_prob: 0.5})

    #Print out training step number and accuracy
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            state: grids_batch, sentence_vec: sentences_batch, rewards_actual: rewards_batch, keep_prob: 0.5})

        print("step %d, training accuracy %g"%(i, train_accuracy))
        steps.append(i)
        accuracy_arr.append(train_accuracy)

'''
Plot the training accuracy.
'''
import matplotlib.pyplot as plt
plt.plot(steps, accuracy_arr)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.savefig('train_results.png')



#saver = tf.train.Saver()
#saver.save(sess, 'trained-model')
    
'''
Print the results on the test data.
'''
test_accuracy = accuracy.eval(feed_dict={
    state: test_grids, sentence_vec: test_sentences, rewards_actual: test_rewards, keep_prob: 1.0})
fp_update_op.eval(feed_dict={
    state: test_grids, sentence_vec: test_sentences, rewards_actual: test_rewards, keep_prob: 1.0})
fn_update_op.eval(feed_dict={
    state: test_grids, sentence_vec: test_sentences, rewards_actual: test_rewards, keep_prob: 1.0})
total_positives = true_positives.eval(feed_dict={
    state: test_grids, sentence_vec: test_sentences, rewards_actual: test_rewards, keep_prob: 1.0})
total_negatives = test_grids.size - total_positives

num_false_positives = false_positives.eval()
num_false_negatives = false_negatives.eval()

#print(num_false_positives)
#print(num_false_negatives)

fp_rate_test = float(num_false_positives) / total_negatives
fn_rate_test = float(num_false_negatives) / total_positives


print ""
print ""
print ""
print("test accuracy %g"%(test_accuracy))
print("false positive rate %f"%(fp_rate_test))
print("false negative rate %f"%(fn_rate_test))

for i in xrange(10):

    j = randint(0, len(test_rewards))

    grid = test_grids[j].reshape((1, 784))
    sentence = test_sentences[j].reshape((1, 4))
    actual_reward = test_rewards[j].reshape((1, 784))

    predicted_reward = rewards_pred_flat.eval(feed_dict={
        state: grid, sentence_vec: sentence, rewards_actual: actual_reward, keep_prob: 1.0})

    visualize_results(actual_reward, predicted_reward, i, "exp2/reward")