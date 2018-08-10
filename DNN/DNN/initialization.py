import numpy as np

def initialize(layers_dims, initialization):
	"""
	初始化参数W，b，这里提供了四种不同的初始化方法
	:param layers_dims: [n_x, n_h1, n_h2, ..., n_hl-1, n_y]，对于L层神经网络，len(layers_dims) = L+1.
	:param initialization: select from "zero", "random", "he"
	:return: parameters_initialization
	"""
	parameters = {}
	# 这里的L相当于神经网络的实际层数+1，因为有A0
	L = len(layers_dims)

	if initialization == "zero":
		for i in range(1, L):
			parameters['W' + str(i)] = np.zeros((layers_dims[i], layers_dims[i - 1]))
			parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))

	elif initialization == "random":
		for i in range(1, L):
			parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1])
			parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))

	elif initialization == "he":
		for i in range(1, L):
			parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2 / layers_dims[i - 1])
			parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))

	elif initialization == "sqrt":
		for i in range(1, L):
			parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(1 / layers_dims[i - 1])
			parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))

	else:
		print("Input wrong initialization...")
		exit(1)

	return parameters


