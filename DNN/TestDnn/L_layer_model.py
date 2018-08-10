from DNN import initialization, forward_propagation, cal_cost, backward_propagation, update_parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):

	costs = []

	parameters = initialization.initialize(layers_dims, 'he')

	for i in range(0, num_iterations):
		AL, caches = forward_propagation.forward(X, parameters)

		costs.append(cal_cost.cal_cost(AL, Y))

		grads = backward_propagation.L_backward(AL, Y, caches)

		parameters = update_parameters.update_para(parameters, grads, learning_rate)

		if print_cost and i % 100 == 0:
			print('Cost after iteration %i: %f' % (i, costs[i]))

	return parameters