import tensorflow as tf
import numpy as np
from TensorFlow_exercise import dataset, exercise
from tensorflow.python.framework import ops
from matplotlib import pyplot as plt

def create_placeholders(n_x, n_y):
	"""
	create the placeholders for the tensorflow session
	:param n_x: size of an image vector: 64 * 64 * 3 = 12288
	:param n_y: number of classes
	:return: X(placeholder for the data input), Y(placeholder for the input labels)
	"""
	X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
	Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')

	return X, Y


def initialize_parameters():
	"""
	初始化参数
	:return: parameters: dict
	"""
	W1 = tf.get_variable('W1', [25, 12288], initializer=tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())
	W2 = tf.get_variable('W2', [12, 25], initializer=tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable('b2', [12, 1], initializer=tf.zeros_initializer())
	W3 = tf.get_variable('W3', [6, 12], initializer=tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable('b3', [6, 1], initializer=tf.zeros_initializer())

	parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

	return parameters


def forward_propagation(X, parameters):
	"""
	:param X: input dataset placeholder. shape(n_x, m)
	:param parameters:
	:return: Z[L]
	"""
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']

	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)

	return Z3


def compute_cost(ZL, Y):
	"""
	:param ZL: output of forward propagation
	:param Y: true labels
	:return: cost
	"""
	logits = tf.transpose(ZL)
	labels = tf.transpose(Y)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

	return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True):

	# to be able to rerun the model without overwriting tf variables
	ops.reset_default_graph()

	(n_x, m) = X_train.shape
	n_y = Y_train.shape[0]
	costs = []

	X, Y = create_placeholders(n_x, n_y)
	parameters = initialize_parameters()
	Z3 = forward_propagation(X, parameters)
	cost = compute_cost(Z3, Y)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for epoch in range(num_epochs):
			epoch_cost = 0
			num_minibatches = int(m / minibatch_size)
			minibatches = exercise.random_mini_batches(X_train, Y_train, mini_batch_size=minibatch_size)

			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				_, minibatch_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
				epoch_cost += minibatch_cost / num_minibatches

			if print_cost and epoch % 100 == 0:
				print('Cost after epoch %i: %f' % (epoch, epoch_cost))
			if print_cost and epoch % 5 == 0:
				costs.append(epoch_cost)

		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("learning rate = "+str(learning_rate))
		plt.show()

		parameters = session.run(parameters)
		print("Parameters have been trained!")

		correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

		return parameters
