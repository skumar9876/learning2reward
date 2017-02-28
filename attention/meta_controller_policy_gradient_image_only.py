import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

from environment import World
import plotting

#from utils import *

'''
Network definition functions.
'''
def weight_variable(shape):

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

            sentence_size = 4

            # Placeholder for the image input in a given iteration
            self.image = tf.placeholder(tf.float32, [None, 10, 10], name='image')
            self.image_flat = tf.reshape(self.image, [-1, 10*10])

            # Placeholder for the sentence in a given iteration
            self.sentence = tf.placeholder(tf.float32, [None, sentence_size], name='sentence')


            ####################
            # Action Variable  # 
            ####################

            self.chosen_room = tf.placeholder(tf.int32, [None, 2], name='chosen_room')

            ###################
            # Target Variable #
            ###################

            self.target = tf.placeholder(tf.float32, [None], name='target')

            ##################
            #     Network    #
            ##################

            # First convolutional layer
            self.W_conv1 = weight_variable([3, 3, 1, 32])
            self.b_conv1 = bias_variable([32])

            self.image_reshaped = tf.reshape(self.image, [-1, 10, 10, 1])
            self.h_conv1 = tf.nn.relu(conv2d(self.image_reshaped, self.W_conv1) + self.b_conv1)
            self.h_pool1 = self.h_conv1
            #self.h_pool1 = max_pool_2x2(self.h_conv1)


            # Second convolutional layer
            self.W_conv2 = weight_variable([3, 3, 32, 64])
            self.b_conv2 = bias_variable([64])
            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)

            self.h_pool2 = max_pool_2x2(self.h_conv2)

            #print self.h_pool2.get_shape()

            self.h_pool2_reshaped = tf.reshape(self.h_pool2, [-1, 25*64])


            image_hidden_layer_size = 25*64

            # Second hidden layer
            self.W_fc2 = weight_variable([image_hidden_layer_size, 4])
            #self.W_fc2 = weight_variable([sentence_hidden_layer_size, 4])
            self.b_fc2 = bias_variable([4])

            self.h_fc2 = (tf.matmul(self.h_pool2_reshaped, self.W_fc2) + self.b_fc2) #ADDED TANH HERE!!!
            #self.h_fc2 = (tf.matmul(self.h_fc_sent, self.W_fc2) + self.b_fc2) 
 
            self.room_vals = tf.reshape(self.h_fc2, [-1, 4])

            self.room_probs = tf.squeeze(tf.nn.softmax(self.room_vals))
            self.picked_room_prob = tf.gather_nd(self.room_probs, self.chosen_room)

            # Loss and train op
            self.loss = - (tf.reduce_sum( tf.multiply( tf.log(self.picked_room_prob), self.target ) ))

            # Computed for debugging purposes
            self.log_prob = tf.log(self.picked_room_prob)
            self.intermediate_loss = tf.multiply (tf.log(self.picked_room_prob), self.target)

            self.probs = self.room_probs * tf.log(self.room_probs)
            self.prob_loss = -tf.reduce_sum(self.room_probs * tf.log(self.room_probs), 1)

            self.room_entropy = -tf.reduce_mean(tf.reduce_sum(self.room_probs * tf.log(self.room_probs), 1), 0)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())



    
    def predict(self, input_image, input_sentence, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.image: input_image, self.sentence: input_sentence}


        room_probs = sess.run(self.room_probs, feed_dict)

        return room_probs

    def update(self, input_image, input_sentence, target, chosen_room, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.image: input_image, self.sentence: input_sentence, self.target: target, self.chosen_room: chosen_room}
        

        _, loss, room_entropy = sess.run([self.train_op, self.loss, self.room_entropy], feed_dict)

        return loss, room_entropy

class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.001, scope="policy_estimator"):
        with tf.variable_scope(scope):
            ###################
            # State Variables #
            ###################

            sentence_size = 4

            # Placeholder for the image input in a given iteration
            self.image = tf.placeholder(tf.float32, [1, 10, 10], name='image')
            self.image_flat = tf.reshape(self.image, [-1, 10*10])

            # Placeholder for the sentence in a given iteration
            self.sentence = tf.placeholder(tf.float32, [1, sentence_size], name='sentence')

            ###################
            # Target Variable #
            ###################

            self.target = tf.placeholder(tf.float32, [1], name='target')

            ##################
            #     Network    #
            ##################

            # First convolutional layer
            self.W_conv1 = weight_variable([3, 3, 1, 32])
            self.b_conv1 = bias_variable([32])

            self.image_reshaped = tf.reshape(self.image, [-1, 10, 10, 1])
            self.h_conv1 = tf.nn.relu(conv2d(self.image_reshaped, self.W_conv1) + self.b_conv1)
            self.h_pool1 = max_pool_2x2(self.h_conv1)


            # Second convolutional layer
            self.W_conv2 = weight_variable([3, 3, 32, 64])
            self.b_conv2 = bias_variable([64])

            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)

            self.h_pool2 = max_pool_2x2(self.h_conv2)

            self.h_pool2_reshaped = tf.reshape(self.h_conv2, [-1, 25*64])


            image_hidden_layer_size = 25*64

            # Second hidden layer
            self.W_fc2 = weight_variable([image_hidden_layer_size, 1])
            self.b_fc2 = bias_variable([1])

            self.value_estimate = tf.matmul(self.h_pool2_reshaped, self.W_fc2) + self.b_fc2

            self.loss = tf.reduce_sum(tf.squared_difference(self.value_estimate, self.target))


            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    
    def predict(self, input_image, input_sentence, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.image: input_image, self.sentence: input_sentence}
        value_estimate = sess.run(self.value_estimate, feed_dict)

        return value_estimate

    def update(self, input_image, input_sentence, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.image: input_image, self.sentence: input_sentence, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)


        return loss



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
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes), 
        loss_arr=np.zeros(num_episodes), 
        loss_entropy_arr=np.zeros(num_episodes), 
        action_entropy_arr=np.zeros(num_episodes), 
        room_entropy_arr=np.zeros(num_episodes))   
    
    Transition = collections.namedtuple("Transition", ["image", "sentence", "room_selection", "next_image", "next_sentence", "done"])

    total_reward = 0
    total_time = 0
    
    for i_episode in range(num_episodes):     
        
        episode = []

        chosen_rooms = []

        images = []
        sentences = []

        reward_arr = []
        baselines = []

        # Reset the environment
        image, target_sentence = env.reset_no_attention()
        print ""
        print "Episode: " + str(i_episode)
        print "target sentence: " + str(target_sentence)
        
        # One step in the environment
        for t in itertools.count():


            image = np.array([image])

            
            # Take a step
            room_probs = estimator_policy.predict(image, target_sentence)

            chosen_room = np.random.choice(np.arange(len(room_probs)), p=room_probs)

            chosen_rooms.append(chosen_room)
            images.append(image)
            sentences.append(target_sentence)

            next_image, next_target_sentence, reward, done = env.step_no_attention(chosen_room)

            reward_arr.append(reward)
            total_reward += reward
            total_time += 1


            if estimator_value != None:
                baselines.append(estimator_value.predict(image, target_sentence))
            else:
                baselines.append(float(total_reward) / total_time)

            if i_episode == num_episodes - 1 or i_episode >= num_episodes - 10 or i_episode <= 9:
                #print "room probs:"
                print room_probs
            
            # Keep track of the transition
            episode.append(Transition(
              image=image, sentence=target_sentence, room_selection=chosen_room, next_image=next_image, next_sentence=next_target_sentence, done=done))
            
            # Update statistics
            # stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            

            if done:
                break
                
            image = next_image



        # Perform update for this episode
        images = np.array(images)
        images = images.reshape((len(images), 10, 10))

        sentences = np.array(sentences)
        sentences = sentences.reshape(len(sentences), 4)

        chosen_rooms_new = []
        for i in xrange(len(chosen_rooms)):
            chosen_rooms_new.append([i, chosen_rooms[i]])
        chosen_rooms = chosen_rooms_new

        chosen_rooms = np.array(chosen_rooms)
        chosen_rooms = chosen_rooms.reshape(len(chosen_rooms), 2)

        stats.episode_rewards[i_episode] = env.episode_reward()


        total_reward_arr = np.zeros(len(reward_arr))
        reinforce_target_arr = np.zeros(len(reward_arr))
        actor_critic_target_arr = np.zeros(len(reward_arr))

        for index in xrange(len(total_reward_arr)):
            for index2 in range(index, len(reward_arr)):
                total_reward_arr[index] += reward_arr[index2]


        if i_episode >= num_episodes - 4:
            print "baselines:"
            print baselines
            print ""
            print "total reward:"
            print total_reward_arr
            print ""

        if estimator_value != None:

            for i in xrange(len(images)):
                # Update estimator value
                estimator_value.update([images[i]], [sentences[i]], [total_reward_arr[i]])


            # Set target to be the advantage
            for i in xrange(len(baselines)):
                reinforce_target_arr[i] = total_reward_arr[i] - baselines[i]


            #for i in xrange(len(baselines) - 1):
            #    actor_critic_target_arr[i] = reward_arr[i] + 0.95 * baselines[i+1] - baselines[i]

        else:
            # Set target to be the advantage
            for i in xrange(len(baselines)):
                reinforce_target_arr[i] = total_reward_arr[i] - baselines[i]

        # Update the policy estimator

        '''
        episode_loss = 0
        episode_loss_entropy = 0
        for i in xrange(len(images)):
            loss, loss_entropy = estimator_policy.update([images[i]], [sentences[i]], [target_arr[i]], [[chosen_rooms[i][1]]])
            episode_loss += loss
            episode_loss_entropy += loss_entropy
        '''

        episode_loss, episode_loss_entropy = estimator_policy.update(images, sentences, reinforce_target_arr, chosen_rooms)

        stats.loss_arr[i_episode] = episode_loss
        stats.loss_entropy_arr[i_episode] = episode_loss_entropy
    
    return stats



tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
estimator_value = ValueEstimator()

fixed=False
if fixed == True:
    fixed_str = 'fixed_conv'
else:
    fixed_str = 'not_fixed_conv'

env = World(fixed=fixed)

num_episodes = 500
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    stats = reinforce(env, policy_estimator, num_episodes, estimator_value=None)

import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_lengths)
fig1.savefig('no_attention2/' + fixed_str + '/episode_lengths.png', dpi=fig1.dpi)

fig2 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_rewards)
fig2.savefig('no_attention2/' + fixed_str + '/episode_rewards.png', dpi=fig2.dpi)

fig3 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.loss_arr)
fig3.savefig('no_attention2/' + fixed_str + '/loss.png', dpi=fig3.dpi)

fig4 = plt.figure()
#print stats.loss_entropy_arr
plt.scatter(np.arange(num_episodes), stats.loss_entropy_arr)
fig4.savefig('no_attention2/' + fixed_str + '/room_entropy.png', dpi=fig4.dpi)