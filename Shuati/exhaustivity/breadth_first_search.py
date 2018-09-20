# 迷宫的最短路径问题，n*m的迷宫由通道和墙壁组成，每一步可以向邻接的上下左右四格的通道移动，求出从起点到终点所需的最小步数
def maze(sx: int, sy: int, gx: int, gy: int, maze: list):
	# 迷宫的大小
	n, m = len(maze), len(maze[0])

	# 四个方向移动的向量
	dx = [1, 0, -1, 0]
	dy = [0, 1, 0, -1]

	# 到各个位置的最短距离
	dist = []
	for i in range(n):
		dist.append([float('inf')] * m)

	# 将起点加入到队列中，并把该点的距离设置为0
	que = [(sx, sy)]
	dist[sx][sy] = 0

	# 不断循环直到队列长度为0
	while que:
		# 从队列中取出第一个位置
		(x, y) = que.pop(0)
		# 如果取出的位置是终点，则结束搜索
		if (x, y) == (gx, gy):
			break

		# 分别移动四个方向，得到下一个位置
		for i in range(4):
			nx, ny = x + dx[i], y + dy[i]
			# 判断新的点是否可以移动，以及是否已经访问过
			if 0 <= nx < n and 0 <= ny < m and maze[nx][ny] == 1 and dist[nx][ny] == float('inf'):
				# 可以移动的话，加入到队列，并且到该位置的距离确定为之前的距离+1
				que.append((nx, ny))
				dist[nx][ny] = dist[x][y] + 1

	return dist[gx][gy]


# 宽度优先搜索同深度优先搜索一样，会生成所有能够遍历到的状态。适合求取最短路径的问题。
if __name__ == '__main__':
	a = [[0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
		 [1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
		 [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
		 [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
		 [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
		 [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
		 [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
		 [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
		 [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
		 [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]]
	print(maze(0, 1, 9, 8, a))