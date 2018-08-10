import numpy as np
import sklearn
import sklearn.datasets
from matplotlib import pyplot as plt

def load_dataset(show=False):
	"""
	:return: train_X, train_Y, test_X, test_Y
	"""
	train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=0.05)
	test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=0.05)

	if show==True:
		plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
		plt.show()

	# (300, 2) to (2, 300)
	train_X = train_X.T
	train_Y = train_Y.reshape([1, train_X.shape[1]])
	test_X = test_X.T
	test_Y = test_Y.reshape([1, test_X.shape[1]])

	return train_X, train_Y, test_X, test_Y

# if __name__=='__main__':
# 	train_X, train_Y, test_X, test_Y = load_dataset(True)
# 	print('end')
