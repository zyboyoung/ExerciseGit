# 从数字三角形的顶部出发到底边，使得路径上所有数字之和最大，每一步只能向下或向右下走
# input: N(行数)
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
def func_recursion(n):
	# 使用递归的方法，深度遍历每条路径，存在大量重复计算
	lists = []
	for i in range(n):
		lists.append([int(x) for x in input().split(' ')])

	def max_sum(i, j):
		if i == n - 1:
			return lists[i][j]
		x = max_sum(i + 1, j)
		y = max_sum(i + 1, j + 1)
		return max(x, y) + lists[i][j]

	return max_sum(0, 0)


@caltime
def func_recursion_pro(n):
	# 记忆递归，把每次的值都保存起来，下次用到时直接使用
	lists = []
	for i in range(n):
		lists.append([int(x) for x in input().split(' ')])

	# 建立列表用于存储每个位置上max_sum的值
	store_max_sum = []
	for i in range(n):
		store_max_sum.append([-1] * (i + 1))

	def max_sum(i, j):
		if store_max_sum[i][j] != -1:
			return store_max_sum[i][j]
		if i == n - 1:
			return lists[i][j]
		else:
			x = max_sum(i + 1, j)
			y = max_sum(i + 1, j + 1)
			store_max_sum[i][j] = max(x, y) + lists[i][j]
		return store_max_sum[i][j]

	return max_sum(0, 0)


@caltime
def func_dp_1(n):
	# 自底向上，分别从最后一层开始计算最大值，直到顶层
	lists = []
	for i in range(n):
		lists.append([int(x) for x in input().split(' ')])

	# 建立列表用于存储每个位置上max_sum的值
	store_max_sum = []
	for i in range(n):
		store_max_sum.append([-1] * (i + 1))

	# 复制最后一列
	for j in range(n):
		store_max_sum[n - 1][j] = lists[n - 1][j]

	for i in range(n - 2, -1, -1):
		for j in range(i + 1):
			store_max_sum[i][j] = max(store_max_sum[i + 1][j], store_max_sum[i + 1][j + 1]) + lists[i][j]

	return store_max_sum[0][0]


@caltime
def func_dp_2(n):
	# 空间优化，只用一维列表存储
	lists = []
	for i in range(n):
		lists.append([int(x) for x in input().split(' ')])

	# 存储最后一行的值
	store_max_sum = lists[-1]

	for i in range(n - 2, -1, -1):
		for j in range(i + 1):
			store_max_sum[j] = max(store_max_sum[j], store_max_sum[j + 1]) + lists[i][j]

	return store_max_sum[0]


if __name__ == '__main__':
	print(func_dp_2(10))
