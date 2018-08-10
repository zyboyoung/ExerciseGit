import tensorflow as tf
import numpy as np
from TensorFlow_exercise import dataset, exercise, model
import time

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = dataset.load_dataset()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

# Convert training and test labels to one hot matrices
Y_train = exercise.convert_to_one_hot(Y_train_orig, 6)
Y_test = exercise.convert_to_one_hot(Y_test_orig, 6)

if __name__=='__main__':
	start_time = time.time()
	parameters = model.model(X_train, Y_train, X_test, Y_test)
	end_time = time.time()
	print('Time cost: '+str(end_time-start_time)+' s')
