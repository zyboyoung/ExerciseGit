# 直线上有N个点，点i的位置是Xi，从中选择若干个，加上标记。对每一个点，其距离为R以内的区域里必须有带有标记的点，目标是为尽可能少的点添加标记
def mark_points(x: list, r: int) -> int:
	n = len(x)
	count, i = 0, 0
	dist = 0
	while i < n:
		if x[i] <= dist or x[i] - dist >= r:
			i += 1
			continue
		else:
			dist = x[i] + r
			count += 1
			i += 1
	return count


# 从最左边开始考虑，给距离为R以内的最远的点标记，对于标记了的点右侧相距超过R的下一个点，继续在其右侧R距离以内最远的点添加标记
if __name__ == '__main__':
	print(mark_points([1, 7, 15, 20, 30, 50], 10))
