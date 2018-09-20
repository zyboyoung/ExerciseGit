# 有n种硬币，面值分别为V1,V2,...,Vn元，每种有无限多。给定非负整数S，能够选用多少硬币，使得面值之和恰好为S元，输出硬币数目的最小值和最大值。
import time
from functools import wraps


def caltime(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print('-' * 8)
		print('start time: ', time.asctime(time.localtime(start)))
		print('end time:   ', time.asctime(time.localtime(end)))
		print('-' * 8)
		print(func.__name__, end - start, 's')
		print('-' * 8)
		return result

	return wrapper


@caltime
def func(n, s):
	# 读取每种硬币的面值
	# values = [int(i) for i in input().split(',')]
	# min, max = [0], [0]
	# def dp():
	# 	for i in range(s):
	# 		for j in range(n):
	# 			if i >= values[j]:
	# 				if min[i - values[j]] < min[i]:
	pass
