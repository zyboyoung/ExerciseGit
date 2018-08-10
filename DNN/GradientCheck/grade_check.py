import numpy as np

def sigmoid(X):
	s = 1 / (1 + np.exp(-X))
	return s


def relu(X):
	s = np.maximum(0, X)
	return s


def forward_propagation(X, Y, parameters):

	m = X.shape[1]
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']

	Z1 = np.dot(W1, X) + b1
	A1 = relu(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = relu(Z2)
	Z3 = np.dot(W3, A2) + b3
	A3 = sigmoid(Z3)

	logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
	cost_l = 1. / m * np.sum(logprobs)
	cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

	return cost_l, cache


def backward_propagation(X, Y, cache):

	m = X.shape[1]
	(Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

	dZ3 = A3 - Y
	dW3 = np.dot(dZ3, A2.T) / m
	db3 = np.sum(dZ3, axis=1, keepdims=True) / m

	dA2 = np.dot(W3.T, dZ3)
	dZ2 = np.multiply(dA2, np.int64(A2 > 0))
	dW2 = np.dot(dZ2, A1.T) / m
	db2 = np.sum(dZ2, axis=1, keepdims=True) / m

	dA1 = np.dot(W2.T, dZ2)
	dZ1 = np.multiply(dA1, np.int64(A1 > 0))
	dW1 = np.dot(dZ1, X.T) / m
	db1 = np.sum(dZ1, axis=1, keepdims=True) / m

	gradients = {
			"dZ3": dZ3, "dW3": dW3, "db3": db3,
			"dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
			"dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1
	}

	return gradients


def dictionary_to_vector(parameters):
	# 将原来字典类型的参数按顺序整合成一维向量
	# keys中存储的是['W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'b1', 'b1', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'b2', 'b2', 'b2', 'W3', 'W3', 'W3', 'b3']纵向排列
	# return: 将W与b整合起来的参数theta

	keys = []
	count = 0
	for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
		column_vector = parameters[key].reshape((-1, 1))
		# column_vector.shape[0]表示Wi或者bi的维度
		keys.append([key] * column_vector.shape[0])

		if count == 0:
			theta = column_vector
		else:
			theta = np.concatenate((theta, column_vector), axis=0)
		count += 1

	return theta, keys


def gradients_to_vector(gradients):
	count = 0
	for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
		column_vector = gradients[key].reshape((-1, 1))

		if count == 0:
			theta = column_vector
		else:
			theta = np.concatenate((theta, column_vector), axis=0)
		count += 1

	return theta


def vector_to_dictionary(theta):
	'''
	将一维列向量转化成字典类型的参数，从而可以进行正向传播，计算cost
	:param theta:
	:return: parameters
	'''
	parameters = {}
	parameters["W1"] = theta[:20].reshape((5, 4))
	parameters["b1"] = theta[20:25].reshape((5, 1))
	parameters["W2"] = theta[25:40].reshape((3, 5))
	parameters["b2"] = theta[40:43].reshape((3, 1))
	parameters["W3"] = theta[43:46].reshape((1, 3))
	parameters["b3"] = theta[46:47].reshape((1, 1))

	return parameters


def gradient_check(parameters, gradients, X, Y, epsilon=1e-7):
	'''
	:param parameters: 由W, b组合起来的字典类型参数
	:param gradients: 字典类型的梯度
	:param X: input
	:param Y: label
	:param epsilon: 极小值
	:return: difference(由导数定义计算的theta梯度，与实际计算的theta梯度之间的误差)
	'''
	parameters_values, _ = dictionary_to_vector(parameters)
	grad = gradients_to_vector(gradients)
	num_parameters = parameters_values.shape[0]

	J_minus = np.zeros((num_parameters, 1))
	J_plus = np.zeros((num_parameters, 1))
	grad_approx = np.zeros((num_parameters, 1))

	# 迭代每一个参数
	for i in range(num_parameters):
		theta_plus = parameters_values.copy()
		theta_plus[i][0] += epsilon
		J_plus[i], _ = forward_propagation(X, Y, vector_to_dictionary(theta_plus))

		theta_minus = parameters_values.copy()
		theta_minus[i][0] -= epsilon

		J_minus[i], _ = forward_propagation(X, Y, vector_to_dictionary(theta_minus))

		# 利用双边导数计算grad_approx
		grad_approx[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

	# 计算difference
	numerator = np.linalg.norm(grad_approx - grad)
	denominator = np.linalg.norm(grad_approx) + np.linalg.norm(grad)
	difference = numerator / denominator

	if difference > 2e-7:
		print('There is a mistake in the backward propagation! difference = ' + str(difference))
	else:
		print('The backward propagation works perfectly fine! difference = ' + str(difference))

	return difference