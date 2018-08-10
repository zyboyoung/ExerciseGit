import tensorflow as tf
from time import time
import numpy as np
from CNN import dataset
from tensorflow.python.framework import ops
from matplotlib import pyplot as plt


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = dataset.load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = dataset.convert_to_one_hot(Y_train_orig, 6).T
Y_test = dataset.convert_to_one_hot(Y_test_orig, 6).T


def create_placeholders(n_H0, n_W0, n_C0, n_y):
	"""
	create the placeholders for tensorflow session
	None用于后面的batch_size
	:param n_H0:
	:param n_W0:
	:param n_C0:
	:param n_y: number of classes
	:return: X-placeholder for the data input, [None, n_H0, n_W0, n_C0]. Y-placeholder for the input labels, [None, n_y]
	"""
	X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
	Y = tf.placeholder(tf.float32, [None, n_y])

	return X, Y


def initialize_parameters():
	"""
	Initializes weight parameters. W1: [4, 4, 3, 8] W2: [2, 2, 8, 16]
	:return: parameters
	"""
	W1 = tf.get_variable('W1', [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

	parameters = {'W1': W1, 'W2': W2}

	return parameters


def forward_propagation(X, parameters):
	"""
	CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
	:param X: input dataset placeholder, of shape (input size, number of examples)
	:param parameters:
	:return: Z3-the output of the last LINEAR unit
	"""
	W1 = parameters['W1']
	W2 = parameters['W2']

	Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
	A1 = tf.nn.relu(Z1)
	P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

	Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
	A2 = tf.nn.relu(Z2)
	P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

	P2 = tf.contrib.layers.flatten(P2)
	Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=6, activation_fn=None)

	return Z3


def compute_cost(Z3, Y):
	"""
	:param Z3: output of forward propagation, of shape (num_outputs, number of examples)
	:param Y: "true" labels vector placeholder
	:return: cost
	"""
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

	return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=100, minibatch_size=64, print_cost=True):
	"""
	Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
	:param X_train: (None, 64, 64, 3)
	:param Y_train: (None, n_y = 6)
	:param X_test: (None, 64, 64, 3)
	:param Y_test: (None, n_y = 6)
	:param learning_rate:
	:param num_epochs:
	:param minibatch_size:
	:param print_cost:
	:return: train_accuracy, test_accuracy, parameters
	"""
	ops.reset_default_graph()
	(m, n_H0, n_W0, n_C0) = X_train.shape
	n_y = Y_train.shape[1]
	costs = []

	# Create Placeholders of the correct shape
	X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

	parameters = initialize_parameters()

	# Forward propagation: Build the forward propagation in the tensorflow graph
	Z3 = forward_propagation(X, parameters)

	cost = compute_cost(Z3, Y)

	# Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	# Initialize all the variables globally
	init = tf.global_variables_initializer()

	# Start the session to compute the tensorflow graph
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost = 0.
			num_minibatches = int(m / minibatch_size)
			minibatches = dataset.random_mini_batches(X_train, Y_train, minibatch_size)

			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				_, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
				minibatch_cost += temp_cost / num_minibatches

			if print_cost == True and epoch % 5 == 0:
				print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(minibatch_cost)

		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

		predict_op = tf.argmax(Z3, 1)
		correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(accuracy)
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
		test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)

		return train_accuracy, test_accuracy, parameters


if __name__ == '__main__':
	start_time = time()

	_, _, parameters = model(X_train, Y_train, X_test, Y_test)

	end_time = time()
	print('------cost time: ' + str(end_time - start_time) + ' s ------')
