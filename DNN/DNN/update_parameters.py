def update_para(parameters, grads, learning_rate):
	L = len(parameters) // 2
	for i in range(L):
		parameters['W' + str(i+1)] -= grads['dW' + str(i+1)] * learning_rate
		parameters['b' + str(i+1)] -= grads['db' + str(i+1)] * learning_rate
	return parameters
