from TestDnn import dataset
from TestDnn import two_layer_model, L_layer_model
from DNN import predict, model
import time

train_x_orig, train_y, test_x_orig, test_y, classes = dataset.load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

n_x = 12288
n_h = 7
n_y = 1
layer_dims_2 = (n_x, n_h, n_y)

layers_dims_L = (12288, 20, 7, 5, 1)

if __name__ == '__main__':

	start_time = time.time()

	layers_DNN = 'L'

	if layers_DNN == 2:
		parameters = two_layer_model.two_layer_model(train_x, train_y, layer_dims_2, num_iterations=2500, print_cost=True)

	elif layers_DNN == 'L':
		# parameters = L_layer_model.L_layer_model(train_x, train_y, layers_dims_L, learning_rate = 0.0075, num_iterations=2500, print_cost=True)
		parameters = model.model(train_x, train_y, hidden_layers_dims=[20, 7, 5, 1], learning_rate=0.0075, num_iterations=2500, initialization_method='sqrt')

	else:
		print('Input wrong layers_DNN')
		exit(1)

	print("On the train set:")
	predictions_train = predict.prediction(train_x, train_y, parameters)
	print("On the test set:")
	predictions_test = predict.prediction(test_x, test_y, parameters)

	end_time = time.time()
	print('[ cost_time: ' + str(end_time - start_time) + ' seconds ]')
