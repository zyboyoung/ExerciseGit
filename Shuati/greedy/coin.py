# 给出各种硬币的面值和每种硬币的数量，输出对于给定目标至少需要多少个硬币
def coin(values: list, counts: list, target: int) -> int:
	count = 0
	i = len(values) - 1
	while target > 0:
		# 直接取最大面值的硬币
		t = min(int(target / values[i]), counts[i])
		target -= t * values[i]
		print(values[i], t)
		i -= 1
		count += t

	return count


if __name__ == '__main__':
	values = [1, 5, 10, 50, 100, 500]
	counts = [3, 2, 1, 3, 0, 2]
	print(coin(values, counts, 620))
