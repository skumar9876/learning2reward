import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

from environment import World
import plotting

from utils import *


class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
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

            self.target = tf.placeholder(dtype=tf.float32, name='target')

            ##################
            #     Network    #
            ##################

            hidden_layer_size = 20

            # Hidden layer for the image input
            self.W_fc1 = weight_variable([10*10, hidden_layer_size])
            self.b_fc1 = bias_variable([hidden_layer_size])
            self.h_fc1 = tf.tanh(tf.matmul(self.image_flat, self.W_fc1) + self.b_fc1) #ADDED SIGMOID HERE

            # Concatenate the sentence
            self.h_fc1_concatenated = tf.concat_v2([self.h_fc1, self.sentence], 1)

            # Second hidden layer
            self.W_fc2 = weight_variable([hidden_layer_size + sentence_size, 4])
            self.b_fc2 = bias_variable([4])

            self.h_fc2 = tf.tanh(tf.matmul(self.h_fc1_concatenated, self.W_fc2) + self.b_fc2) #ADDED SIGMOID HERE!!!
 
            self.room_vals = tf.reshape(self.h_fc2, [-1, 4])

            self.room_probs = tf.squeeze(tf.nn.softmax(self.room_vals))
            self.picked_room_prob = tf.gather_nd(self.room_probs, self.chosen_room)

            # Loss and train op
            self.loss = - (tf.reduce_sum(tf.log(self.picked_room_prob))) * self.target

            self.probs = self.room_probs * tf.log(self.room_probs)
            self.prob_loss = -tf.reduce_sum(self.room_probs * tf.log(self.room_probs), 1)

            self.loss_entropy = -tf.reduce_mean(tf.reduce_sum(self.room_probs * tf.log(self.room_probs), 1), 0)

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

        #print self.room_probs.eval(feed_dict)
        #print ""
        #print self.probs.eval(feed_dict)
        #print ""
        #print self.prob_loss.eval(feed_dict)

        _, loss, loss_entropy = sess.run([self.train_op, self.loss, self.loss_entropy], feed_dict)

        return loss, loss_entropy


def reinforce(env, estimator_policy, num_episodes):
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
    
    for i_episode in range(num_episodes):     
        
        episode = []

        chosen_rooms = []

        images = []
        sentences = []


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

            if i_episode == num_episodes - 1 or i_episode == num_episodes - 2 or i_episode == 0:
                print room_probs

            chosen_room = np.random.choice(np.arange(len(room_probs)), p=room_probs)

            chosen_rooms.append(chosen_room)
            images.append(image)
            sentences.append(target_sentence)

            next_image, next_target_sentence, done = env.step_no_attention(chosen_room)
            
            # Keep track of the transition
            episode.append(Transition(
              image=image, sentence=target_sentence, room_selection=chosen_room, next_image=next_image, next_sentence=next_target_sentence, done=done))
            
            # Update statistics
            # stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # Print out which step we're on, useful for debugging.
            # print("\rStep {} @ Episode {}/{} ({})".format(
            #        t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))
            # sys.stdout.flush()

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

        target = env.episode_reward()
        stats.episode_rewards[i_episode] = target

        loss, loss_entropy = estimator_policy.update(images, sentences, target, chosen_rooms)

        stats.loss_arr[i_episode] = loss
        stats.loss_entropy_arr[i_episode] = loss_entropy
    
    return stats



tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()

fixed=True
if fixed == True:
    fixed_str = 'fixed'
else:
    fixed_str = 'not_fixed'

env = World(fixed=fixed)

num_episodes = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    stats = reinforce(env, policy_estimator, num_episodes)

import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_lengths)
fig1.savefig('no_attention/' + fixed_str + '/episode_lengths.png', dpi=fig1.dpi)

fig2 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_rewards)
fig2.savefig('no_attention/' + fixed_str + '/episode_rewards.png', dpi=fig2.dpi)

fig3 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.loss_arr)
fig3.savefig('no_attention/' + fixed_str + '/loss.png', dpi=fig3.dpi)

fig4 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.loss_entropy_arr)
fig4.savefig('no_attention/' + fixed_str + '/loss_entropy.png', dpi=fig4.dpi)