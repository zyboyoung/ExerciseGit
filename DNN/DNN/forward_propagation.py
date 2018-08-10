import numpy as np

def sigmoid(Z):
	A = 1 / (1 + np.exp(-Z))
	return A, Z

def relu(Z):
	A = np.maximum(0, Z)
	return A, Z

def activation(A_pre, W, b, activate_func):
	"""
	一次激活函数，计算当前层的A
	:param A_pre:
	:param W:
	:param b:
	:param activate_func: 可选sigmoid，relu
	:return: 当前层的A, cache: ((A_pre, W, b), Z)
	"""
	Z = np.dot(W, A_pre) + b
	forward_cache = (A_pre, W, b)

	if activate_func == 'sigmoid':
		A, activate_cache = sigmoid(Z)
	elif activate_func == 'relu':
		A, activate_cache = relu(Z)

	cache = (forward_cache, activate_cache)

	return A, cache

def forward(X, parameters):
	"""
	L层神经网络的正向传播，前L-1层使用relu激活函数，最后一层使用sigmoid激活函数
	:return: AL（即Yhat），caches[cache1, cache2, ..., cacheL]
	"""
	caches = []
	A = X

	# 这里的L是神经网络的实际层数
	L = len(parameters) // 2

	# 对1到L-1层进行ReLU激活函数
	for i in range(1, L):
		A_pre = A
		A, cache = activation(A_pre, parameters['W' + str(i)], parameters['b' + str(i)], activate_func = 'relu')
		caches.append(cache)

	# 对第L层进行sigmoid激活函数
	(AL, cache) = activation(A, parameters['W' + str(L)], parameters['b' + str(L)], activate_func = 'sigmoid')
	caches.append(cache)

	return AL, caches

def forward_propagation_with_dropout(X, parameters, keep_prob):
	"""
	:param X:
	:param parameters:
	:param keep_prob:
	:return: AL, cache[Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3]
	"""

	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']

	Z1 = np.dot(W1, X) + b1
	A1 = relu(Z1)[0]
	D1 = np.random.rand(np.shape(A1)[0], np.shape(A1)[1])
	D1 = D1 < keep_prob
	A1 = np.multiply(A1, D1)
	A1 = A1 / keep_prob

	Z2 = np.dot(W2, A1) + b2
	A2 = relu(Z2)[0]
	D2 = np.random.rand(np.shape(A2)[0], np.shape(A2)[1])
	D2 = D2 < keep_prob
	A2 = np.multiply(A2, D2)
	A2 = A2 / keep_prob

	Z3 = np.dot(W3, A2) + b3
	A3 = sigmoid(Z3)[0]

	cache = [Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3]

	return A3, cache
