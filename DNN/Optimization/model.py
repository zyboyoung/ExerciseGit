from Optimization import optimization_methods
from DNN import initialization, forward_propagation, cal_cost, backward_propagation, update_parameters
from matplotlib import pyplot as plt

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
	"""
	优化算法模型
	:param optimizer: 梯度下降的三种方法：gd, Momentum, Adam
	:param mini_batch_size:
	:param beta: 用于Momentum的超参数
	:param beta1: Adam中用于Momentum的超参数
	:param beta2: Adam中用于RMSprop的超参数
	:param epsilon: 避免出现除以0的超参数
	:param num_epochs: 迭代次数
	:param print_cost:
	:return: parameters
	"""

	L = len(layers_dims)
	costs = []
	t = 0
	parameters = initialization.initialize(layers_dims, initialization='he')

	# initialize optimizer
	if optimizer=='gd':
		pass
	elif optimizer=='Momentum':
		v = optimization_methods.initialize_velocity(parameters)
	elif optimizer=='Adam':
		v, s = optimization_methods.initialize_adam(parameters)

	for i in range(num_epochs):
		mini_batches = optimization_methods.random_mini_batches(X, Y, mini_batch_size)

		for mini_batch in mini_batches:
			(mini_batch_X, mini_batch_Y) = mini_batch

			AL, caches = forward_propagation.forward(mini_batch_X, parameters)
			cost = cal_cost.cal_cost(AL, mini_batch_Y)
			grads = backward_propagation.L_backward(AL, mini_batch_Y, caches)

			if optimizer=='gd':
				parameters = update_parameters.update_para(parameters, grads, learning_rate)
			elif optimizer=='Momentum':
				parameters, v = optimization_methods.update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
			elif optimizer=='Adam':
				t += 1
				parameters, v, s = optimization_methods.update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

		if print_cost and i % 1000 == 0:
			print('Cost after epoch %i: %f' %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)

	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('epochs (per 100)')
	plt.title('learning rate = ' + str(learning_rate))
	plt.show()

	return parameters

