from matplotlib import pyplot as plt
from DNN import predict
import numpy as np

def plot_decision_boundary(model, X, Y):
	"""
	适合X中只有两个特征的分类
	:param model:
	:param X:
	:param Y:
	:return:
	"""
	# Set min and max values and give it some padding
	x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
	y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
	h = 0.01
	# Generate a grid of points with distance h between them
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Predict the function value for the whole grid
	Z = model(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), cmap=plt.cm.Spectral)
	plt.show()


def plt_show(initialization, X, Y, parameters):
	plt.title("Model with " + initialization + " initialization")
	axes = plt.gca()
	axes.set_xlim([-1.5, 1.5])
	axes.set_ylim([-1.5, 1.5])
	plot_decision_boundary(lambda x: predict.predict_plot(parameters, x.T), X, Y)