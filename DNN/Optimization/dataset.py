import sklearn.datasets
from matplotlib import pyplot as plt

def load_dataset():
	train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
	plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
	train_X = train_X.T
	train_Y = train_Y.reshape((1, train_Y.shape[0]))
	plt.show()

	return train_X, train_Y

