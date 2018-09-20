# 01背包问题：有n个重量和价值分别为wi，vi的物品，从这些物品中挑选出总重量不超过W的物品，目标是价值总和最大。
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


# 第一种方法：对于每个物品都进行两种选择。搜索深度是n，最坏需要O（2^^n）复杂度。
@caltime
def bag_1(w: list, v: list, w_total: int) -> int:
	n = len(w)

	def choice(i, target):
		if i == n:
			return 0
		elif w[i] > target:
			return choice(i + 1, target)
		else:
			return max(choice(i + 1, target), choice(i + 1, target - w[i]) + v[i])

	return choice(0, w_total)


# 第二种方法：记忆化搜索。第一次计算时记录下每一步的结果，以免后续的重复计算。复杂度降低为O（n*w_total）
@caltime
def bag_2(w, v, w_total):
	# 建立列表，存储第一次计算时的结果
	n = len(w)
	res_store = []
	for i in range(n):
		res_store.append([float('inf')] * (w_total + 1))

	def choice(i, target):
		if i < n and res_store[i][target] != float('inf'):
			return res_store[i][target]
		if i == n:
			return 0
		elif w[i] > target:
			return choice(i + 1, target)
		else:
			res_store[i][target] = max(choice(i + 1, target), choice(i + 1, target - w[i]) + v[i])
			return res_store[i][target]

	return choice(0, w_total)


if __name__ == '__main__':
	w = [2, 1, 3, 2, 2, 3, 2, 1, 4, 2, 2, 1, 3, 2, 2, 3, 2, 1, 4, 2]
	v = [3, 2, 4, 2, 3, 4, 1, 2, 3, 1, 3, 2, 4, 2, 3, 4, 1, 2, 3, 1]
	print(bag_1(w, v, 25))
