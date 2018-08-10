from DNN import initialization, forward_propagation, cal_cost, backward_propagation, update_parameters
import numpy as np

def two_layer_model(X, Y, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = True):

	grads = {}
	costs = []

	parameters = initialization.initialize(layer_dims, 'sqrt')

	for i in range(num_iterations):

		W1 = parameters["W1"]
		b1 = parameters["b1"]
		W2 = parameters["W2"]
		b2 = parameters["b2"]

		A1, cache1 = forward_propagation.activation(X, W1, b1, 'relu')
		A2, cache2 = forward_propagation.activation(A1, W2, b2, 'sigmoid')
		costs.append(cal_cost.cal_cost(A2, Y))

		dA2 = - np.divide(Y, A2) + np.divide((1 - Y), 1 - A2)
		dA1, dW2, db2 = backward_propagation.backward(dA2, cache2, 'sigmoid')
		dA0, dW1, db1 = backward_propagation.backward(dA1, cache1, 'relu')

		grads['dW1'] = dW1
		grads['db1'] = db1
		grads['dW2'] = dW2
		grads['db2'] = db2

		parameters = update_parameters.update_para(parameters, grads, learning_rate)

		if print_cost and i % 100 == 0:
			print('Cost after iteration {}: {}'.format(i, np.squeeze(costs[i])))

	return parameters
