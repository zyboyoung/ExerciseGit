# train：600 * 64 * 64 * 3, test: 150 * 64 * 64 * 3
import h5py
import numpy as np


def load_dataset():
	train_dataset = h5py.File(r'D:\教程\深度学习\deeplearning.ai-master\4_Convolution Neural Networks\Week2\Keras Tutorial\datasets\train_happy.h5', 'r')
	train_set_x_orig = np.array(train_dataset["train_set_x"][:])
	train_set_y_orig = np.array(train_dataset["train_set_y"][:])

	test_dataset = h5py.File(r'D:\教程\深度学习\deeplearning.ai-master\4_Convolution Neural Networks\Week2\Keras Tutorial\datasets\test_happy.h5', 'r')
	test_set_x_orig = np.array(test_dataset["test_set_x"][:])
	test_set_y_orig = np.array(test_dataset["test_set_y"][:])

	classes = np.array(test_dataset["list_classes"][:])

	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def normalize_vectors():
	X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

	X_train = X_train_orig / 255
	X_test = X_test_orig / 255

	Y_train = Y_train_orig.T
	Y_test = Y_test_orig.T

	return X_train, X_test, Y_train, Y_test, classes
