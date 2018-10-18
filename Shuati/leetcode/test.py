import sys


n, m, k = [int(i) for i in sys.stdin.readline().split(' ')]
i, item_time = 0, []
v1, v2 = [], []
while i < n:
	a, b, c = sys.stdin.readline().split()
	item_time.append(int(a))
	v1.append(int(b))
	v2.append(int(c))
	i += 1
j, graph = 0, []
while j < m:
	graph.append([int(i) for i in sys.stdin.readline().split()])
	j += 1
print(graph)
