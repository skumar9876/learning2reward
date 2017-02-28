"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

from state_network import StateProcessor
from utils import weight_variable, bias_variable, loglikelihood
from config import Config



rnn_cell = tf.nn.rnn_cell
seq2seq = tf.nn.seq2seq


config = Config()
n_steps = config.step

sampled_action_arr = []
sampled_room_arr = []


def get_next_input(output, i):
  """
  Takes in output of LSTM, determines the 
  new attention from the LSTM action and 
  predicted sentence, and outputs the processed
  output of the new attention and target sentence.
  """

  action_vals = tf.slice(output, [0, 0], [1, 4])
  room_vals = tf.slice(output, [0, 4], [1, 4])

  action_probs = tf.squeeze(tf.nn.softmax(action_vals))
  picked_action = tf.argmax(action_probs, 1) # Pick action greedily -> is this correct?
  #picked_action = np.random.choice(np.arange(len(action_probs)), p=action_probs) #get index of picked action
  picked_action_prob = tf.gather(action_probs, picked_action)

  room_probs = tf.squeeze(tf.nn.softmax(room_vals))
  #picked_room = np.random.choice(np.arange(len(room_probs)), p=room_probs) #get index of picked room
  picked_room = tf.argmax(room_probs, 1)
  picked_room_prob = tf.gather(room_probs, picked_room)

  next_attention = env.step(picked_action=picked_action, picked_room=picked_room)

  processed_state = state_processor(next_attention, self.target_sentence)
  
  sampled_action_arr.append(picked_action_prob)
  sampled_room_arr.append(picked_room_prob)

  return processed_state


# Build the state processor
with tf.variable_scope('state_processor'):
  state_processor = StateProcessor(config)


# number of examples
N = tf.shape(images_ph)[0]
init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
init_glimpse = gl(init_loc)


# Placeholder for the starting image input in a given episode
init_image = tf.placeholder(tf.float32, [1, 5, 5], "image")

# Placeholder for the target sentence in a given episode
target_sentence = tf.placeholder(tf.float32, [1,5], "sentence")

# Get first processed output
processed_output = state_processor(init_image, target_sentence)

# Core network.
lstm_cell = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
# Initial state of the LSTM memory.
init_state = tf.zeros([1, lstm.state_size])

inputs = [processed_output]
inputs.extend([0] * (config.num_iterations))

outputs, _ = seq2seq.rnn_decoder(
    inputs, init_state, lstm_cell, loop_function=get_next_input)

'''
# Time independent baselines
with tf.variable_scope('baseline'):
  w_baseline = weight_variable((config.cell_output_size, 1))
  b_baseline = bias_variable((1,))
baselines = []
for t, output in enumerate(outputs[1:]):
  baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
  baseline_t = tf.squeeze(baseline_t)
  baselines.append(baseline_t)
baselines = tf.pack(baselines)  # [timesteps, batch_sz]
baselines = tf.transpose(baselines)  # [batch_sz, timesteps]
'''
'''
# Take the last step only.
output = outputs[-1]
# Build classification network.
with tf.variable_scope('cls'):
  w_logit = weight_variable((config.cell_output_size, config.num_classes))
  b_logit = bias_variable((config.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits)



# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_ph)
xent = tf.reduce_mean(xent)
pred_labels = tf.argmax(logits, 1)
# 0/1 reward.
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std)
advs = rewards - tf.stop_gradient(baselines)
logllratio = tf.reduce_mean(logll * advs)
reward = tf.reduce_mean(reward)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()
# hybrid loss
loss = -logllratio + xent + baselines_mse  # `-` for minimize
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
'''

# Get the episode reward from the environment
episode_reward = env.episode_reward()



log_prob = - (tf.log(sampled_action_arr) + tf.log(sampled_room_arr))




# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = mnist.train.num_examples // config.batch_size
starter_learning_rate = config.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch,
    0.97,
    staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for i in xrange(n_steps):
    images, labels = mnist.train.next_batch(config.batch_size)
    # duplicate M times, see Eqn (2)
    images = np.tile(images, [config.M, 1])
    labels = np.tile(labels, [config.M])
    loc_net.samping = True
    adv_val, baselines_mse_val, xent_val, logllratio_val, \
        reward_val, loss_val, lr_val, _ = sess.run(
            [advs, baselines_mse, xent, logllratio,
             reward, loss, learning_rate, train_op],
            feed_dict={
                images_ph: images,
                labels_ph: labels
            })
    if i and i % 100 == 0:
      logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
      logging.info(
          'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
              i, reward_val, loss_val, xent_val))
      logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
          logllratio_val, baselines_mse_val))

    if i and i % training_steps_per_epoch == 0:
      # Evaluation
      for dataset in [mnist.validation, mnist.test]:
        steps_per_epoch = dataset.num_examples // config.eval_batch_size
        correct_cnt = 0
        num_samples = steps_per_epoch * config.batch_size
        loc_net.sampling = True
        for test_step in xrange(steps_per_epoch):
          images, labels = dataset.next_batch(config.batch_size)
          labels_bak = labels
          # Duplicate M times
          images = np.tile(images, [config.M, 1])
          labels = np.tile(labels, [config.M])
          softmax_val = sess.run(softmax,
                                 feed_dict={
                                     images_ph: images,
                                     labels_ph: labels
                                 })
          softmax_val = np.reshape(softmax_val,
                                   [config.M, -1, config.num_classes])
          softmax_val = np.mean(softmax_val, 0)
          pred_labels_val = np.argmax(softmax_val, 1)
          pred_labels_val = pred_labels_val.flatten()
          correct_cnt += np.sum(pred_labels_val == labels_bak)
        acc = correct_cnt / num_samples
        if dataset == mnist.validation:
          logging.info('valid accuracy = {}'.format(acc))
        else:
          logging.info('test accuracy = {}'.format(acc))