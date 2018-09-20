# 部分和问题，判断是否可以从data中选出若干数，使他们的和恰好为k
def part_sum(data: list, k: int) -> bool:
	# 从第一个元素开始，决定加或不加，在所有数都决定后判断和是否是k即可。复杂度：O（2^n）
	def dfs(i, s):
		# 表示从前i项得到的和为s
		if i == len(data) - 1:
			# 如果第i项已经是最后一项，那么判断和sum是否与k相等
			return s == k
		if dfs(i + 1, s):
			# 如果不是最后一项，不加上第i + 1项的情况
			return True
		if dfs(i + 1, s + data[i + 1]):
			# 不是最后一项，加上第i + 1项的情况
			return True

		return False

	if dfs(0, 0):
		return True
	else:
		return False


# lake_counting问题，大小为N*M的园子，雨后积水，八连通的积水认为是连接在一起的，求出园子里总共有多少水洼
def count_lake(field: list) -> int:
	# 输入field是一个二维矩阵（N*M），其中的元素为‘1’和‘0’，前者表示有积水，后者表示空地。时间复杂度O（n*m）
	n, m = len(field), len(field[0])

	def dfs(x: int, y: int) -> None:
		field[x][y] = 0
		# 将所在位置由1替换为0
		for dx in [-1, 0, 1]:
			for dy in [-1, 0, 1]:
				# 找出所有与所在位置是八连通的点
				nx, ny = x + dx, y + dy
				if 0 <= nx < n and 0 <= ny < m and field[nx][ny] == 1:
					# 判断新的点是否有积水，有的话继续找其八连通的点
					dfs(nx, ny)
		return

	count = 0
	for i in range(n):
		for j in range(m):
			if field[i][j] == 1:
				# 找出一共深度搜索了几次
				dfs(i, j)
				count += 1

	return count


# 深度优先搜索：从某个状态开始，不断地转移状态直到无法转移，然后回退到前一步的状态，继续转移到其它状态，不断重复，直到找到最终的解
if __name__ == '__main__':
	print(count_lake([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1]]))
