from functools import wraps
import time
import numpy as np
import pandas as pd
import jieba
import wordcloud
from scipy.misc import imread
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns


# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['SimHei']


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
def show_novel():
	# 导入停用词，转换为列表格式
	stop_list = []
	with open('materials/停用词.txt', 'r') as f:
		for line in f.readlines():
			stop_list.append(line.strip())

	# 导入小说原文
	with open('materials/全职高手.txt', 'r', encoding='gbk') as f:
		text = f.read()

	# 导入词库字典，编码方式必须是utf-8
	jieba.load_userdict('materials/词库字典.txt')

	# 分词功能的函数，完成分词
	def txt_cut(f):
		return [w for w in jieba.cut(f) if w not in stop_list and len(w) > 1]

	textcut = txt_cut(text)

	# 对词频进行简单统计
	word_count = pd.Series(textcut).value_counts().sort_values(ascending=False)[0:20]

	# 画出柱状图并保存
	fig = plt.figure(figsize=(15, 8))
	x = word_count.index.tolist()
	y = word_count.values.tolist()
	sns.barplot(x, y, palette='BuPu_r')
	plt.title('词频Top20')
	plt.ylabel('count')
	sns.despine(bottom=True)
	plt.savefig('词频统计.png', dpi=400)
	plt.show()

	# 实例化一个词云类，添加分词
	fig_cloud = plt.figure(figsize=(15, 5))
	# font_path导入中文字体，mask设置词云图案的形状
	cloud = wordcloud.WordCloud(
		font_path='materials/simkai.ttf',
		mask=imread('materials/test.jpg'),
		mode='RGBA',
		background_color=None
	).generate(' '.join(textcut))

	# 对词的颜色做美化
	img = imread('materials/color.jpg')
	cloud_colors = wordcloud.ImageColorGenerator(np.array(img))
	cloud.recolor(color_func=cloud_colors)

	# 调用matplotlib接口
	plt.imshow(cloud)
	plt.axis('off')
	plt.savefig('wordcloud.png', dpi=400)
	plt.show()


if __name__ == '__main__':
	show_novel()
