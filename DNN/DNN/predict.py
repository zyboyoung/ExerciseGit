import numpy as np
from DNN import forward_propagation

def prediction(X, Y, parameters):
	m = X.shape[1]
	p = np.zeros((1, m))

	probas, cache = forward_propagation.forward(X, parameters)

	for i in range(0, probas.shape[1]):
		if probas[0, i] > 0.5:
			p[0, i] = 1
		else:
			p[0, i] = 0

	print('Accuracy: ' + str(np.sum((p == Y)/m)))

	return p

def predict_plot(parameters, X):
	"""
	Used for plotting decision boundary.
	:param parameters:
	:param X:
	:return:
	"""
	AL, cache = forward_propagation.forward(X, parameters)
	predictions = (AL > 0.5)
	return predictions