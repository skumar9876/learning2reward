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
    
    #learning_rate = 0.01
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            ###################
            # State Variables #
            ###################

            sentence_size = 4

            # Placeholder for the image input in a given iteration
            self.image = tf.placeholder(tf.float32, [None, 5, 5], name='image')
            self.image_flat = tf.reshape(self.image, [-1, 5*5])

            # Placeholder for the sentence in a given iteration
            self.sentence = tf.placeholder(tf.float32, [None, sentence_size], name='sentence')

            # LSTM 
            self.lstm_size = 9

            self.lstm_unit = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, state_is_tuple=False)


            #self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, self.lstm_size])
            #self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, self.lstm_size])
            
            #self.lstm_state_input = tf.nn.rnn_cell.LSTMStateTuple(self.initial_lstm_state0,
            #                                                  self.initial_lstm_state1)


            # Placeholder for the current lstm state
            self.lstm_state_input = tf.placeholder(tf.float32, [1, self.lstm_size*2], name='lstm_state')

            ####################
            # Action Variables # 
            ####################

            self.chosen_action = tf.placeholder(tf.int32, [None, 2], name='chosen_action')
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
            self.W_fc1 = weight_variable([5*5, hidden_layer_size])
            self.b_fc1 = bias_variable([hidden_layer_size])
            self.h_fc1 = tf.tanh(tf.matmul(self.image_flat, self.W_fc1) + self.b_fc1) #Added tanh activation

            # Concatenate the sentence
            self.h_fc1_concatenated = tf.concat_v2([self.h_fc1, self.sentence], 1)

            # Second hidden layer
            self.W_fc2 = weight_variable([hidden_layer_size + sentence_size, hidden_layer_size])
            self.b_fc2 = bias_variable([hidden_layer_size])

            self.h_fc2 = tf.tanh(tf.matmul(self.h_fc1_concatenated, self.W_fc2) + self.b_fc2) #Added tanh activation

            self.h_fc2_reshaped = tf.reshape(self.h_fc2, [1,-1,hidden_layer_size])


            self.step_size = tf.placeholder(tf.int32)
            # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
            # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
            # Unrolling step size is applied via self.step_size placeholder.
            # When forward propagating, step_size is 1.
            # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])

            self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm_unit,
                                                            self.h_fc2_reshaped,
                                                            initial_state = self.lstm_state_input,
                                                            sequence_length = self.step_size,
                                                            time_major = False,
                                                            scope = scope)

            self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1,9])

            # CHECK THESE TWO LINES TO MAKE SURE I AM GETTING THE RIGHT VALUES
            self.action_vals = tf.slice(self.lstm_outputs, [0, 0], [self.step_size, 5])
            self.room_vals = tf.slice(self.lstm_outputs, [0, 5], [self.step_size, 4])


            self.action_probs = tf.squeeze(tf.nn.softmax(self.action_vals))
            self.picked_action_prob = tf.gather_nd(self.action_probs, self.chosen_action)

            self.room_probs = tf.squeeze(tf.nn.softmax(self.room_vals))
            self.picked_room_prob = tf.gather_nd(self.room_probs, self.chosen_room)

            # Loss and train op
            self.loss = - (tf.reduce_sum(tf.log(self.picked_action_prob)) + tf.reduce_sum(tf.log(self.picked_room_prob))) * self.target

            self.action_entropy = -tf.reduce_mean(tf.reduce_sum(self.action_probs * tf.log(self.action_probs), 1), 0)
            self.room_entropy = -tf.reduce_mean(tf.reduce_sum(self.room_probs * tf.log(self.room_probs), 1), 0)

            action_tensor = tf.log(self.picked_action_prob)
            room_tensor = tf.log(self.picked_room_prob)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    
    def predict(self, input_image, input_sentence, lstm_state_input, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.image: input_image, self.sentence: input_sentence, self.lstm_state_input: lstm_state_input, self.step_size: 1}

        action_probs = sess.run(self.action_probs, feed_dict)
        room_probs = sess.run(self.room_probs, feed_dict)
        lstm_state = sess.run(self.lstm_state, feed_dict)


        return action_probs, room_probs, lstm_state

    def update(self, input_image, input_sentence, target, chosen_action, chosen_room, lstm_state_input, step_size, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.image: input_image, self.sentence: input_sentence, self.target: target, self.chosen_action: chosen_action, self.chosen_room: chosen_room, 
        self.lstm_state_input: lstm_state_input, self.step_size: step_size}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        lstm_outputs = self.lstm_outputs.eval(feed_dict)
        action_vals = self.action_vals.eval(feed_dict)
        action_probs = self.action_probs.eval(feed_dict)
        chosen_action = self.chosen_action.eval(feed_dict)
        picked_action_prob = self.picked_action_prob.eval(feed_dict)

        picked_room_prob = self.picked_room_prob.eval(feed_dict)
        loss = self.loss.eval(feed_dict)
        action_entropy = self.action_entropy.eval(feed_dict)
        room_entropy = self.room_entropy.eval(feed_dict)

        '''
        print lstm_outputs
        print ""
        print action_vals
        print ""
        print action_probs
        print len(action_probs)
        print ""
        print chosen_action
        print len(chosen_action)
        print ""
        print picked_action_prob
        print len(picked_action_prob)
        print ""
        '''
        
        return loss, action_entropy, room_entropy



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
    
    Transition = collections.namedtuple("Transition", ["image", "sentence", "move_action", "room_selection", "next_image", "next_sentence", "done"])
    
    for i_episode in range(num_episodes):
        
        
        episode = []


        chosen_rooms = []
        chosen_actions = []

        images = []
        sentences = []

        # Initial state of the LSTM memory.
        initial_state = lstm_state = np.zeros([1, 18])


        # Reset the environment
        image, target_sentence = env.reset()
        print ""
        print "Episode: " + str(i_episode)
        print "target sentence: " + str(target_sentence)
        
        # One step in the environment
        for t in itertools.count():


            image = np.array([image])

            
            # Take a step
            action_probs, room_probs, lstm_state = estimator_policy.predict(image, target_sentence, lstm_state)

            if i_episode == num_episodes - 1 or i_episode == num_episodes - 2 or i_episode == 0:
                print action_probs
                print room_probs

            #print action_probs

            chosen_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            chosen_room = np.random.choice(np.arange(len(room_probs)), p=room_probs)

            chosen_actions.append(chosen_action)
            chosen_rooms.append(chosen_room)

            images.append(image)
            sentences.append(target_sentence)

            next_image, next_target_sentence, done = env.step(chosen_room, chosen_action)
            
            # Keep track of the transition
            episode.append(Transition(
              image=image, sentence=target_sentence, move_action=chosen_action, room_selection=chosen_room, next_image=next_image, next_sentence=next_target_sentence, done=done))
            
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

        #print chosen_actions
        #print chosen_rooms

        # Perform update for this episode
        images = np.array(images)
        images = images.reshape((len(images), 5, 5))

        sentences = np.array(sentences)
        sentences = sentences.reshape(len(sentences), 4)

        chosen_rooms_new = []
        for i in xrange(len(chosen_rooms)):
            chosen_rooms_new.append([i, chosen_rooms[i]])
        chosen_rooms = chosen_rooms_new

        chosen_rooms = np.array(chosen_rooms)
        chosen_rooms = chosen_rooms.reshape(len(chosen_rooms), 2)

        chosen_actions_new = []
        for i in xrange(len(chosen_actions)):
            chosen_actions_new.append([i, chosen_actions[i]])
        chosen_actions = chosen_actions_new

        chosen_actions = np.array(chosen_actions)
        chosen_actions = chosen_actions.reshape(len(chosen_actions), 2)

        lstm_initial_state = np.zeros([1, 18])

        target = env.episode_reward()
        stats.episode_rewards[i_episode] = target

        loss, action_entropy, room_entropy = estimator_policy.update(images, sentences, target, chosen_actions, chosen_rooms, lstm_initial_state, len(episode))

        stats.loss_arr[i_episode] = loss
        stats.action_entropy_arr[i_episode] = action_entropy
        stats.room_entropy_arr[i_episode] = room_entropy
    
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
fig1.savefig('attention/' + fixed_str + '/episode_lengths.png', dpi=fig1.dpi)

fig2 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_rewards)
fig2.savefig('attention/' + fixed_str + '/episode_rewards.png', dpi=fig2.dpi)

fig3 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.loss_arr)
fig3.savefig('attention/' + fixed_str + '/loss.png', dpi=fig3.dpi)

fig4 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.action_entropy_arr)
fig4.savefig('attention/' + fixed_str + '/action_entropy.png', dpi=fig4.dpi)

fig5 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.room_entropy_arr)
fig5.savefig('attention/' + fixed_str + '/room_entropy.png', dpi=fig5.dpi)