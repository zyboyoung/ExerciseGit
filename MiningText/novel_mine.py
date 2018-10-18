from functools import wraps
import time
import jieba
import codecs
import jieba.posseg as pseg
import pandas as pd


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
		print(func.__name__, ': ', end - start, 's')
		print('-' * 8)
		return result

	return wrapper


@caltime
def mine():
	# 建立三个容器用于转换数据
	names = {}  # 保存人名和出现的频数
	relationships = {}  # 提取人物关系
	lineNames = []  # 每一行提取的人名，缓存

	# 导入词库和文本，排除词性不为nr，长度大于3小于2的所有词汇
	jieba.load_userdict('materials/词库字典.txt')
	with codecs.open('materials/全职高手.txt', 'r', 'gbk') as f:
		n = 0
		for line in f.readlines():
			n += 1
			print('正在读取第{}行'.format(n))
			poss = pseg.cut(line)
			lineNames.append([])
			for w in poss:
				if w.flag != 'nr' or len(w.word) < 2 or len(w.word) > 3:
					continue
				lineNames[-1].append(w.word)
				if names.get(w.word) is None:
					names[w.word] = 0
					relationships[w.word] = {}
				names[w.word] += 1

	# 遍历lineNames，对每一行出现的人名进行匹配，建立人物关系
	for line in lineNames:
		for name1 in line:
			for name2 in line:
				if name1 == name2:
					continue
				if relationships[name1].get(name2) is None:
					relationships[name1][name2] = 1
				else:
					relationships[name1][name2] += 1

	# 构建两个数组，用于归类点数据和边数据
	node = pd.DataFrame(columns=['Id', 'Label', 'Weight'])
	edge = pd.DataFrame(columns=['Source', 'Target', 'Weight'])

	# 把清洗好的数据添加进数组中
	for name, times in names.items():
		node.loc[len(node)] = [name, name, times]

	for name, edges in relationships.items():
		for v, w in edges.items():
			if w > 3:
				edge.loc[len(edge)] = [name, v, w]

	# 导出数据待用
	edge.to_csv('edge_raw.csv', index=0)
	node.to_csv('node_raw.csv', index=0)


if __name__ == '__main__':
	mine()
