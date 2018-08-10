from GradientCheck import dataset, grade_check

if __name__ == '__main__':
	X, Y, parameters = dataset.gradient_check_n_test_case()

	cost, cache = grade_check.forward_propagation(X, Y, parameters)
	gradients = grade_check.backward_propagation(X, Y, cache)
	difference = grade_check.gradient_check(parameters, gradients, X, Y)