import scipy.io
from matplotlib import pyplot as plt
import numpy as np

def load_2D_dataset(show=False):
	"""

	:return:
	"""
	data = scipy.io.loadmat(r'D:\研究生\教程\深度学习\deeplearning.ai-master\2_Improving Deep Neural Networks Hyperparameter tuning, Regularization and Optimization\Week1\2_Regularization\datasets\data.mat')
	train_x = data['X'].T
	train_y = data['y'].T
	test_x = data['Xval'].T
	test_y = data['yval'].T

	if show:
		plt.scatter(train_x[0, :], train_x[1, :], c=np.squeeze(train_y), s=40, cmap=plt.cm.Spectral)
		plt.show()

	return train_x, train_y, test_x, test_y

if __name__=='__main__':
	train_x, train_y, test_x, test_y = load_2D_dataset(True)
	print(train_x, train_y, test_x, test_y, sep='\n')


