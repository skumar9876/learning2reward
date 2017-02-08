import tensorflow as tf

# Network definition functions
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


sess = tf.InteractiveSession()

# Placeholder for the image input in a given iteration.
image = tf.placeholder(tf.float32, [1, 5, 5])

image_flat = tf.reshape(image, [1, 5*5])

# Placeholder for the sentence in a given iteration
sentence = tf.placeholder(tf.float32, [1, 5])

# Hidden layer for the image input
W_fc1 = weight_variable([5*5, 100])
b_fc1 = bias_variable([100])
h_fc1 = tf.matmul(image_flat, W_fc1) +  b_fc1

# Concatenate the sentence
h_fc1_concatenated = tf.concat_v2([h_fc1, sentence], 1)

# Second hidden layer
W_fc2 = weight_variable([105, 100])
b_fc2 = weight_variable([100])

h_fc2 = tf.matmul(h_fc1_concatenated, W_fc2) + b_fc2


# LSTM unit
lstm_size = 9
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=False)

print lstm.state_size

# Initial state of the LSTM memory.
initial_state = state = tf.zeros([1, lstm.state_size])
#initial_state = state = tf.zeros([1, lstm_size])


# LSTM outputs:
#	1. A sentence instruction of length 4
#	2. An action vector of length 5 
# 		- Actions are stay, left, right, up, down 
output, state = lstm(h_fc2, state)

sess.run(tf.global_variables_initializer())


num_steps = 100

for i in range(num_steps):
	
	attention_output = output.eval(feed_dict={
            image: att, sentence: sent })


	pred_sentence = tf.slice(attention_output, [0, 0], [1, 4])
	pred_action = tf.slice(attention_output, [0, 4],[1, 5])

	

	agent.move(sent)
	loss += #call reward function here -> returns 0 if agent doesn't reach goal here, -1 if agent reaches goal here

	att = #function call here that takes in action as input
	sent = pred_sentence

# How to propogate loss up?
# This train step needs to be defined earlier somehow...
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)




#sess.run(tf.global_variables_initializer())

#print lstm(h_fc2, state)

#output, state = lstm(h_fc2, state)

'''
for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state
'''