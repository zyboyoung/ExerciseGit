import numpy as np


# 第一种：递归
def fibonacci_recursion_tool(n):
	if n <= 0:
		return 0
	elif n == 1:
		return 1
	else:
		return fibonacci_recursion_tool(n - 1) + fibonacci_recursion_tool(n - 2)


def fibonacci_recursion(n):
	result_list = []
	for i in range(1, n + 1):
		result_list.append(fibonacci_recursion_tool(i))
	return result_list


# 第二种：循环
def fibonacci_loop_tool(n):
	a, b = 0, 1
	while n > 0:
		a, b = b, a + b
		n -= -1
	return b


def fibonacci_loop(n):
	result_list = []
	a, b = 0, 1
	while n > 0:
		result_list.append(b)
		a, b = b, a + b
		n -= 1
	return result_list


# 第三种：yield
def fibonacci_yield_tool(n):
	a, b = 0, 1
	while n > 0:
		yield b
		a, b = b, a + b
		n -= 1


def fibonacci_yield(n):
	# return [f for i, f in enumerate(fibonacci_yield_tool(n))]
	return list(fibonacci_yield_tool(n))


# 第四种：矩阵求解：直接求出第n位数值。将原有问题转化为了矩阵运算
def fibonacci_matrix_tool(n):
	base_matrix = np.matrix('1 1; 1 0')
	return pow(base_matrix, n)


def fibonacci_matrix(n):
	result_list = []
	for i in range(0, n):
		result_list.append(np.array(fibonacci_matrix_tool(i))[0][0])
	return result_list
