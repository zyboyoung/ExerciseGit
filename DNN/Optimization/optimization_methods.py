import numpy as np
import math

def random_mini_batches(X, Y, mini_batch_size=64):
	"""
	需要先将原始数据进行随机分散，再按照mini_batch_size进行划分
	:param X: input
	:param Y: label
	:param mini_batch_size: 大小一般为2的幂
	:return: mini_batches: [(mini_batch_X_1, mini_batch_Y_1), (mini_batch_X_2, mini_batch_Y_2), ...]
	"""

	m = X.shape[1]
	mini_batches = []

	# shuffle
	shuffled_m = list(np.random.permutation(m))
	shuffled_X = X[:, shuffled_m]
	shuffled_Y = Y[:, shuffled_m]

	# partition
	# 向下取整获得完整的batches数
	num_complete_batches = math.floor(m / mini_batch_size)
	for i in range(num_complete_batches):
		mini_batch_X = shuffled_X[:, i * mini_batch_size: (i+1) * mini_batch_size]
		mini_batch_Y = shuffled_Y[:, i * mini_batch_size: (i+1) * mini_batch_size]

		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	# 处理剩余部分数据
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_batches* mini_batch_size: m]
		mini_batch_Y = shuffled_Y[:, num_complete_batches* mini_batch_size: m]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches


# Momentum
def initialize_velocity(parameters):
	"""
	初始化权重的velocity
	:param parameters:
	:return: v: {'dWi': , 'dbi': ,}
	"""

	L = len(parameters) // 2
	v = {}

	for i in range(L):
		v['dW' + str(i+1)] = np.zeros(parameters['W' + str(i+1)].shape)
		v['db' + str(i+1)] = np.zeros(parameters['b' + str(i+1)].shape)

	return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
	"""
	使用velocity对原有参数W, b进行更新
	:param parameters:
	:param grads:
	:param v:
	:param beta:
	:param learning_rate:
	:return:
	"""

	L = len(parameters) // 2

	for i in range(L):
		v['dW' + str(i+1)] = beta*v['dW' + str(i+1)] + (1-beta)*grads['dW' + str(i+1)]
		v['db' + str(i+1)] = beta*v['db' + str(i+1)] + (1-beta)*grads['db' + str(i+1)]

		parameters['W' + str(i+1)] = parameters['W' + str(i+1)] - learning_rate*v['dW' + str(i+1)]
		parameters['b' + str(i+1)] = parameters['b' + str(i+1)] - learning_rate*v['db' + str(i+1)]

	return parameters, v


# Adam
def initialize_adam(parameters):
	"""
	:param parameters:
	:return: v: {'dWi': , 'dbi': ,}, s: {'dWi': , 'dbi': ,}
	"""
	L = len(parameters) // 2
	v = {}
	s = {}

	for i in range(L):
		v['dW' + str(i+1)] = np.zeros(parameters['W' + str(i+1)].shape)
		v['db' + str(i+1)] = np.zeros(parameters['b' + str(i+1)].shape)

		s['dW' + str(i+1)] = np.zeros(parameters['W' + str(i+1)].shape)
		s['db' + str(i+1)] = np.zeros(parameters['b' + str(i+1)].shape)

	return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8 ):
	"""
	:param parameters:
	:param grads:
	:param v:
	:param s:
	:param t:
	:param learning_rate:
	:param beta1:
	:param beta2:
	:param epsilon:
	:return:
	"""

	L = len(parameters) // 2
	v_corrected = {}
	s_corrected = {}

	for i in range(L):
		# Momentum
		v['dW' + str(i+1)] = beta1*v['dW' + str(i+1)] + (1-beta1)*grads['dW' + str(i+1)]
		v['db' + str(i+1)] = beta1*v['db' + str(i+1)] + (1-beta1)*grads['db' + str(i+1)]

		v_corrected['dW' + str(i+1)] = v['dW' + str(i+1)] / (1-(beta1**t))
		v_corrected['db' + str(i+1)] = v['db' + str(i+1)] / (1-(beta1**t))

		# RMSprop
		s['dW' + str(i+1)] = beta2*s['dW' + str(i+1)] + (1-beta2)*np.square(grads['dW' + str(i+1)])
		s['db' + str(i+1)] = beta2*s['db' + str(i+1)] + (1-beta2)*np.square(grads['db' + str(i+1)])

		s_corrected['dW' + str(i+1)] = s['dW' + str(i+1)] / (1-(beta2**t))
		s_corrected['db' + str(i+1)] = s['db' + str(i+1)] / (1-(beta2**t))

		# update parameters
		parameters['W' + str(i+1)] -= learning_rate*(np.divide(v_corrected['dW' + str(i+1)], np.sqrt(s_corrected['dW' + str(i+1)] + epsilon)))
		parameters['b' + str(i+1)] -= learning_rate*(np.divide(v_corrected['db' + str(i+1)], np.sqrt(s_corrected['db' + str(i+1)] + epsilon)))

	return parameters, v, s
