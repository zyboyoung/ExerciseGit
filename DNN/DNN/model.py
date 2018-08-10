from DNN import forward_propagation, cal_cost, backward_propagation, update_parameters, initialization
from matplotlib import pyplot as plt

def model(X, Y, hidden_layers_dims, learning_rate=0.3, num_iterations=30000, initialization_method='he', print_cost=True, lambd=0, keep_prob=1):
	"""
	默认为学习率0.3，迭代30000次，每500次输出cost，不进行正则化和dropout
	:param X: input data [n_x, m]
	:param Y: label [1, m]
	:param print_cost: print the cost every 500 iterations
	:param lambd: regulation hyperparameter
	:param keep_prob: probability of keeping a neuron active during drop-out
	:return: parameters learnt by the model
	"""
	grads = {}
	costs = []
	# m = X.shape[1]
	hidden_layers_dims.insert(0, X.shape[0])
	layers_dims = hidden_layers_dims

	parameters = initialization.initialize(layers_dims, initialization_method)

	for i in range(1, num_iterations + 1):
		if keep_prob == 1:
			AL, caches = forward_propagation.forward(X, parameters)
		elif keep_prob < 1:
			AL, cache = forward_propagation.forward_propagation_with_dropout(X, parameters, keep_prob)
		else:
			print('Input wrong keep_prob')
			break

		if lambd == 0:
			cost_l = cal_cost.cal_cost(AL, Y)
		else:
			cost_l = cal_cost.cal_cost_with_regularization(AL, Y, parameters, lambd)

		# lambd==0表示没有进行L正则化，keep_prob==1表示没有进行dropout，这里是为了确保L正则化和dropout没有同时进行
		assert (lambd == 0 or keep_prob == 1)

		# 既没有L正则化，也没有dropout的情况
		if lambd == 0 and keep_prob == 1:
			grads = backward_propagation.L_backward(AL, Y, caches)
		# 只存在L正则化的情况
		elif lambd != 0:
			grads = backward_propagation.L_backward(AL, Y, caches, lambd)
		# 只存在dropout的情况
		elif keep_prob < 1:
			grads = backward_propagation.backward_propagation_with_dropout(X, Y, cache, keep_prob)

		parameters = update_parameters.update_para(parameters, grads, learning_rate)

		if print_cost and i % 1000 == 0:
			print('Cost after iteration {}: {}'.format(i, cost_l))
			costs.append(cost_l)

	if print_cost:
		plt.plot(costs)
		plt.ylabel('cost')
		plt.xlabel('iterations (per thousand)')
		plt.title('Learning rate =' + str(learning_rate))
		plt.show()

	return parameters
