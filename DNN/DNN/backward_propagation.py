import numpy as np

def sigmoid_backward(dA, activate_cache):
	Z = activate_cache
	A = 1 / (1 + np.exp(-Z))
	dZ = dA * A * (1 - A)
	return dZ


def relu_backward(dA, activate_cache):
	Z = activate_cache
	dZ = dA.copy()
	dZ[Z < 0] = 0
	return dZ


def backward(dA, cache, activate_func, m, lambd=0):
	# 一次反向传播
	# input: 当前层的dA, 当前层的cache(正向传播中存储的((A_pre, W, b), Z), 当前层的激活函数
	# output: dA_pre(也就是前一层的dA), 当前层的dW和db
	forward_cache, activate_cache = cache
	A_pre, W, b = forward_cache

	if lambd == 0:
		if activate_func == 'sigmoid':
			dZ = sigmoid_backward(dA, activate_cache)

			dW = np.dot(dZ, A_pre.T) / m
			db = np.sum(dZ, axis=1, keepdims=True) / m
			dA_pre = np.dot(W.T, dZ)

		elif activate_func == 'relu':
			dZ = relu_backward(dA, activate_cache)

			dW = np.dot(dZ, A_pre.T) / m
			db = np.sum(dZ, axis=1, keepdims=True) / m
			dA_pre = np.dot(W.T, dZ)

		else:
			print('activate_func only can be sigmoid or relu')
			exit(1)

	else:
		if activate_func == 'sigmoid':
			dZ = sigmoid_backward(dA, activate_cache)

			dW = np.dot(dZ, A_pre.T) / m + (lambd / m) * W
			db = np.sum(dZ, axis=1, keepdims=True) / m
			dA_pre = np.dot(W.T, dZ)

		elif activate_func == 'relu':
			dZ = relu_backward(dA, activate_cache)

			dW = np.dot(dZ, A_pre.T) / m + (lambd / m) * W
			db = np.sum(dZ, axis=1, keepdims=True) / m
			dA_pre = np.dot(W.T, dZ)

		else:
			print('activate_func only can be sigmoid or relu')
			exit(1)

	return dA_pre, dW, db


def L_backward(AL, Y, caches, lambd=0):
	"""
	反向传播的迭代进行
	:param AL: Yhat
	:param Y:
	:param caches: caches[cache1, cache2, ..., cacheL]，其中cache_l((A_pre, W, b), Z)
	:return:
	"""
	#
	# input：AL, Y, caches[正向传播中存储的((A_pre, W, b), Z]
	# output:
	grads = {}
	L = len(caches)
	Y = Y.reshape(AL.shape)
	m = Y.shape[1]

	dAL = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
	current_cache = caches[L - 1]
	grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = backward(dAL, current_cache, activate_func='sigmoid', m=m, lambd=lambd)

	for i in reversed(range(L - 1)):
		current_cache = caches[i]
		dA_prev, dW, db = backward(grads['dA' + str(i + 2)], current_cache, activate_func= 'relu', m=m, lambd=lambd)
		grads['dA' + str(i + 1)] = dA_prev
		grads['dW' + str(i + 1)] = dW
		grads['db' + str(i + 1)] = db

	return grads


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
	"""
	:param AL:
	:param Y:
	:param cache: [Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3]
	:param keep_prob:
	:return: grads
	"""
	Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3 = cache
	m = X.shape[1]

	# 第三层没有dropout
	dZ3 = A3 - Y
	dW3 = np.dot(dZ3, A2.T) / m
	db3 = np.sum(dZ3, axis=1, keepdims=True) / m

	# 前两层存在dropout
	dA2 = np.dot(W3.T, dZ3)
	dA2 = np.multiply(dA2, D2)
	dA2 = dA2 / keep_prob
	dZ2 = np.multiply(dA2, np.int64(A2 > 0))
	dW2 = np.dot(dZ2, A1.T) / m
	db2 = np.sum(dZ2, axis=1, keepdims=True) / m

	dA1 = np.dot(W2.T, dZ2)
	dA1 = np.multiply(dA1, D1)
	dA1 = dA1 / keep_prob
	dZ1 = np.multiply(dA1, np.int64(A1 > 0))
	dW1 = np.dot(dZ1, X.T) / m
	db1 = np.sum(dZ1, axis=1, keepdims=True) / m

	grads = {
		'dZ3': dZ3, 'dW3': dW3, 'db3': db3,
		'dZ2': dZ2, 'dW2': dW2, 'db2': db2, 'dA2': dA2,
		'dZ1': dZ1, 'dW1': dW1, 'db1': db1, 'dA1': dA1
	}

	return grads