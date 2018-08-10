from Improvement import dataset
from DNN import predict, plot_result, model
import time

if __name__=='__main__':
	start_time = time.time()

	train_X, train_Y, test_X, test_Y = dataset.load_dataset(True)
	hidden_layers_dims = [10, 5, 1]
	initialization_method = 'he'
	parameters = model.model(train_X, train_Y, hidden_layers_dims, learning_rate=0.0075, num_iterations=10000, initialization_method=initialization_method)

	# print(parameters)
	print("On the train set:")
	predictions_train = predict.prediction(train_X, train_Y, parameters)
	print("On the test set:")
	predictions_test = predict.prediction(test_X, test_Y, parameters)

	plot_result.plt_show(initialization_method, train_X, train_Y, parameters)

	end_time = time.time()
	print('[ cost_time: ' + str(end_time - start_time) + ' seconds ]')
