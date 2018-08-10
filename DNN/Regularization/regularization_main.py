from Regularization import dataset
from DNN import predict, plot_result, model
import time

if __name__=='__main__':
	start_time = time.time()

	hidden_layers_dims = [20, 3, 1]
	train_x, train_y, test_x, test_y = dataset.load_2D_dataset(True)
	initialization_method = 'random'
	parameters = model.model(train_x, train_y, hidden_layers_dims, learning_rate=0.3, initialization_method=initialization_method)

	# print(parameters)
	print("On the train set:")
	predictions_train = predict.prediction(train_x, train_y, parameters)
	print("On the test set:")
	predictions_test = predict.prediction(test_x, test_y, parameters)

	plot_result.plt_show(initialization_method, train_x, train_y, parameters)

	end_time = time.time()
	print('[ cost_time: ' + str(end_time - start_time) + 'seconds ]')