from Optimization import optimization_methods, dataset, model
from DNN import predict, plot_result
from matplotlib import pyplot as plt

if __name__=='__main__':
	train_X, train_Y = dataset.load_dataset()
	layers_dims = [train_X.shape[0], 5, 2, 1]
	parameters = model.model(train_X, train_Y, layers_dims, optimizer='Adam')

	predictions = predict.prediction(train_X, train_Y, parameters)
	plt.title("Model with Adam optimization")
	axes = plt.gca()
	axes.set_xlim([-1.5, 2.5])
	axes.set_ylim([-1, 1.5])
	plot_result.plot_decision_boundary(lambda x: predict.predict_plot(parameters, x.T), train_X, train_Y)
