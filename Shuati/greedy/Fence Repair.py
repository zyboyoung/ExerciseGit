# 一块长度为N的木板，切割成给定长度的几块，每次切割的开销是切割成的两块的长度之和，目标是将木板切割完最小的开销
def fence_split(ln: list) -> int:
	n = sum(ln)
	ln.sort(reverse=True)
	res = 0
	for i in range(len(ln) - 1):
		res += n
		n -= ln[i]
	return res


if __name__ == '__main__':
	print(fence_split([8, 5, 8]))
