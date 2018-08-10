import numpy as np

def cal_cost(AL, Y):
	"""
	:param AL: Yhat
	:param Y:
	:return: cost, float
	"""
	m = Y.shape[1]
	cost = -(np.nansum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))) / m
	cost = np.squeeze(cost)
	return cost

def cal_cost_with_regularization(AL, Y, parameters, lambd):
	"""
	在L2正则化下，重新计算cost函数
	:param AL: Yhat
	:param Y: labels vector
	:param parameters:
	:param lambd: L2正则化参数
	:return: cost, float
	"""
	m = Y.shape[1]

	L = len(parameters) // 2
	sum = 0
	for i in range(1, L + 1):
		sum += np.sum(np.square(parameters['W' + str(i)]))

	L2_regularization_cost =  (1 / m) * (lambd / 2) * sum

	cost_part = cal_cost(AL, Y)
	cost = cost_part + L2_regularization_cost

	return cost
