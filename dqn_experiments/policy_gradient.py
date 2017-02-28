import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

from environment import GridWorld
import plotting_pg
import random

'''
from utils import *
'''

'''
Network definition functions.
'''
def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)

    #initial = tf.get_variable(name=name, shape=shape,
    #       initializer=tf.contrib.layers.xavier_initializer())
    #return initial

    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')



class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.00001, scope="policy_estimator"):
        with tf.variable_scope(scope):
            ###################
            # State Variables #
            ###################


            # Placeholder for the image input in a given iteration
            self.image = tf.placeholder(tf.float32, [None, 10, 10], name='image')

            ####################
            # Action Variable  # 
            ####################

            self.chosen_room = tf.placeholder(tf.int32, [None, 1], name='chosen_room')

            ###################
            # Target Variable #
            ###################

            self.target = tf.placeholder(tf.float32, [None], name='target')

            ##################
            #     Network    #
            ##################

            #First convolutional layer
            self.W_conv1 = weight_variable([3, 3, 1, 32], 'conv1')
            self.b_conv1 = bias_variable([32])

            self.image_reshaped = tf.reshape(self.image, [-1, 10, 10, 1])
            self.h_conv1 = tf.nn.relu(conv2d(self.image_reshaped, self.W_conv1) + self.b_conv1)

            #Second convolutional layer
            self.W_conv2 = weight_variable([3, 3, 32, 64], 'conv2')
            self.b_conv2 = bias_variable([64])
            self.h_conv2 = tf.nn.relu(conv2d(self.h_conv1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = max_pool_2x2(self.h_conv2)

            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 25*64])



            hidden_layer_size = 4

            # Hidden layer for the image input
            self.W_fc1 = weight_variable([25*64, hidden_layer_size], 'fc1')
            self.b_fc1 = bias_variable([hidden_layer_size])
            self.h_fc1 = (tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1) 

            '''
            # Second hidden layer
            self.W_fc2 = weight_variable([hidden_layer_size, 4])
            self.b_fc2 = bias_variable([4])

            self.h_fc2 = (tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2) 
            '''
 
            self.room_vals = tf.reshape(self.h_fc1, [-1, 4])
            self.room_probs = tf.squeeze(tf.nn.softmax(self.room_vals))
            #self.room_probs_reshaped = tf.reshape(self.room_probs, [-1, 4])
            self.room_probs_reshaped = self.room_probs


            self.picked_room_prob = tf.gather(self.room_probs_reshaped, self.chosen_room)

            # Loss and train op
            # self.loss = - (tf.reduce_sum(tf.multiply( tf.log(self.picked_room_prob), self.target ) ))
            self.loss = -tf.multiply(tf.log(self.picked_room_prob), self.target)

            # Computed for debugging purposes
            self.log_prob = tf.log(self.picked_room_prob)
            self.intermediate_loss = tf.multiply (tf.log(self.picked_room_prob), self.target)

            self.probs = self.room_probs * tf.log(self.room_probs)
            self.prob_loss = -tf.reduce_sum(self.room_probs_reshaped * tf.log(self.room_probs_reshaped), 1)

            #self.loss_entropy = -tf.reduce_mean(tf.reduce_sum(self.room_probs_reshaped * tf.log(self.room_probs_reshaped), 1), 0)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    
    def predict(self, input_image, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.image: input_image}

        room_probs = sess.run(self.room_probs, feed_dict)

        return room_probs

    def update(self, input_image, target, chosen_room, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.image: input_image, self.target: target, self.chosen_room: chosen_room}

        '''
        print "room probs:"
        print self.room_probs.eval(feed_dict)
        print ""
        print "chosen room:"
        print self.chosen_room.eval(feed_dict)
        print ""
        print "chosen prob:"
        print self.picked_room_prob.eval(feed_dict)
        print ""
        print "log prob:"
        print self.log_prob.eval(feed_dict)
        print "loss:"
        print self.loss.eval(feed_dict)
        print ""
        '''

        '''
        print ""
        print "target:"
        print self.target.eval(feed_dict)
        '''
        

        #_, loss, loss_entropy = sess.run([self.train_op, self.loss, self.loss_entropy], feed_dict)
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        #return loss, loss_entropy
        return loss, 0

class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.001, scope="value_estimator"):
        with tf.variable_scope(scope):
            ###################
            # State Variables #
            ###################


            # Placeholder for the image input in a given iteration
            self.image = tf.placeholder(tf.float32, [None, 10, 10], name='image')


            ###################
            # Target Variable #
            ###################

            self.target = tf.placeholder(tf.float32, [None], name='target')

            ##################
            #     Network    #
            ##################

            #First convolutional layer
            self.W_conv1 = weight_variable([3, 3, 1, 32], 'conv1')
            self.b_conv1 = bias_variable([32])

            self.image_reshaped = tf.reshape(self.image, [-1, 10, 10, 1])
            self.h_conv1 = tf.nn.relu(conv2d(self.image_reshaped, self.W_conv1) + self.b_conv1)

            #Second convolutional layer
            self.W_conv2 = weight_variable([3, 3, 32, 64], 'conv2')
            self.b_conv2 = bias_variable([1])
            self.h_conv2 = tf.nn.relu(conv2d(self.h_conv1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = max_pool_2x2(self.h_conv2)

            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 64*25])



            hidden_layer_size = 1

            # Hidden layer for the image input
            self.W_fc1 = weight_variable([64*5*5, hidden_layer_size], 'fc1')
            self.b_fc1 = bias_variable([hidden_layer_size])
            self.value_estimate = (tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1) 

            self.loss = tf.reduce_sum(tf.squared_difference(self.value_estimate, self.target))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    
    def predict(self, input_image, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.image: input_image}
        value_estimate = sess.run(self.value_estimate, feed_dict)

        return value_estimate

    def update(self, input_image, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.image: input_image, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss

def random_sample():
    return np.randint(0,3)

def prob_sample(probs):
    return np.random.choice(np.arange(len(probs)), p=probs)

def eps_greedy(probs, epsilon):
    num = random.random()
    if num < epsilon:
        return sample(probs)
    else:
        return greedy(probs)

def greedy(probs):
    return np.argmax(probs)


def reinforce(env, estimator_policy, num_episodes, estimator_value=None):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting_pg.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes), 
        loss_arr=np.zeros(num_episodes), 
        loss_entropy_arr=np.zeros(num_episodes), 
        action_entropy_arr=np.zeros(num_episodes), 
        room_entropy_arr=np.zeros(num_episodes))   
    
    Transition = collections.namedtuple("Transition", ["image", "room_selection", "next_image", "done"])

    #epsilon = 1
    #epsilon_decay = 0.99
    
    for i_episode in range(num_episodes):     
        
        
        episode = []

        chosen_rooms = []

        images = []

        reward_arr = []
        baselines = []

        total_reward = 0
        total_time = 0

        # Reset the environment
        image, target_sentence = env.reset()
        print ""
        print "Episode: " + str(i_episode)
        print "target sentence: " + str(target_sentence)

        #epsilon *= epsilon_decay
        #print "Epsilon:"
        #print epsilon
        
        # One step in the environment
        for t in itertools.count():

            image = np.array(image)
            image = [image]
            
            # Take a step
            room_probs = estimator_policy.predict(image)

            #chosen_room = np.random.choice(np.arange(len(room_probs)), p=room_probs)
            chosen_room = prob_sample(room_probs) #eps_greedy(room_probs, epsilon)
            '''
            print "CHOSEN ROOM:"
            print chosen_room
            '''

            #action_dict = {0:'up', 1:'right', 2:'down', 3:'left'}
            #print action_dict[chosen_room]

            chosen_rooms.append(chosen_room)
            #images.append(image)
            images.append(image)


            next_image, reward, done = env.step(chosen_room)

            reward_arr.append(reward)
            total_reward += reward
            total_time += 1

            if estimator_value != None:
                baselines.append(estimator_value.predict(image))
            else:
                baselines.append(float(total_reward) / total_time)

            if i_episode == num_episodes - 1 or i_episode == num_episodes - 2 or i_episode <= 10:
                print room_probs
            
            # Keep track of the transition
            episode.append(Transition(
              image=image, room_selection=chosen_room, next_image=next_image, done=done))
            
            # Update statistics
            stats.episode_lengths[i_episode] = t
            
            if done:
                break
                
            image = next_image



        if i_episode == num_episodes - 1 or i_episode == num_episodes - 2:
            print baselines
            #pass

        # Perform update for this episode
        images = np.array(images)
        images = images.reshape((len(images), 10, 10))


        chosen_rooms_new = []
        for i in xrange(len(chosen_rooms)):
            chosen_rooms_new.append([i, chosen_rooms[i]])
        chosen_rooms = chosen_rooms_new

        chosen_rooms = np.array(chosen_rooms)
        chosen_rooms = chosen_rooms.reshape(len(chosen_rooms), 2)


        #stats.episode_rewards[i_episode] = env.episode_reward()


        target_arr = np.zeros(len(reward_arr))
        for index in xrange(len(target_arr)):
            for index2 in range(index, len(reward_arr)):
                target_arr[index] += reward_arr[index2]

        if estimator_value != None:

            for i in xrange(len(images)):
                # Update estimator value
                estimator_value.update([images[i]], [target_arr[i]])

            # Set target to be the advantage
            for i in xrange(len(baselines)):
                target_arr[i] = target_arr[i] - baselines[i]
        else:
            # Set target to be the advantage
            for i in xrange(len(baselines)):
                target_arr[i] = target_arr[i] - baselines[i]

        # Update the policy estimator

        episode_loss = 0
        episode_loss_entropy = 0
        for i in xrange(len(images)):
            loss, loss_entropy = estimator_policy.update([images[i]], [target_arr[i]], [[chosen_rooms[i][1]]])
            episode_loss += loss
            episode_loss_entropy += loss_entropy

        stats.loss_arr[i_episode] = episode_loss
        stats.loss_entropy_arr[i_episode] = episode_loss_entropy
    
    return stats



tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
estimator_value = ValueEstimator()



env = GridWorld()

num_episodes = 500
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    stats = reinforce(env, policy_estimator, num_episodes, estimator_value=None)

import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_lengths)
fig1.savefig('policy_gradient/episode_lengths.png', dpi=fig1.dpi)

fig2 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_rewards)
fig2.savefig('policy_gradient/episode_rewards.png', dpi=fig2.dpi)

fig3 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.loss_arr)
fig3.savefig('policy_gradient/loss.png', dpi=fig3.dpi)

fig4 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.loss_entropy_arr)
fig4.savefig('policy_gradient/room_entropy.png', dpi=fig4.dpi)