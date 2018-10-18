import pandas as pd
import numpy as np
import scipy.stats as ss


df = pd.read_csv('data/HR.csv', engine='python')
# 一些基本操作，查询
# print(df.mean())	# 求均值
# print(df.median())	# 求中位数
# print(df.quantile(q=0.25))	# 求四分位数
# print(df.mode())	# 求众数
# print(df.std())	# 求标准差
# print(df.var())	# 求方差
# print(df.sum())	# 求和
# print(df.skew())	# 求偏态系数
# print(df.kurt())	# 求峰态系数
# print(df.sample(n=10))	# 随机取样

# 查找一些空值
# sl_s = df['satisfaction_level']
# print(sl_s.isnull())
# print(sl_s[sl_s.isnull()])
# print(df[sl_s.isnull()])
# sl_s = sl_s.dropna()
# print(np.histogram(sl_s.values, bins=np.arange(0.0, 1.1, 0.1)))	# 表现某列值的分布情况，分别展示0~0.1，0.1~0.2...之间各有多少样本

# 查找一些异常值
# le_s = df['last_evaluation']
# print(le_s[le_s > 1])
# q_low = le_s.quantile(q=0.25)
# q_high = le_s.quantile(q=0.75)
# q_interval = q_high - q_low
# k = 1.5
# le_s = le_s[le_s < q_high + k * q_interval][le_s > q_low - k * q_interval]		# 利用四分位数确定上下界，进行异常值的去除

# 查询某列数据的频率（结构分布）
# np_s = df['number_project']
# print(np_s.value_counts())									# 展示该列每个值出现的频数，默认按照从大到小排序
# print(np_s.value_counts(normalize=True))					# 展示该列每个值出现的频率，默认按照从大到小排序
# print(np_s.value_counts(normalize=True).sort_index())		# 按照index重新排序

# 去除异常的离散值
# s_s = df['salary']
# print(s_s.value_counts())					# 找出异常的离散值的取值
# s_s = s_s.where(s_s != 'nme').dropna()	# 去除异常的离散值

# 整体的数据预处理
df = df.dropna(axis=0, how='any')			# axis=0表示删除行，axis=1表示删除列；how='any'表示只要有一个空值就认为满足条件，how='all'表示全为空值时才视为满足条件
df = df[df['last_evaluation'] <= 1][df['salary'] != 'nme'][df['department'] != 'sale']
df.groupby('department').mean()		# 这里按照department属性分组，其它属性显示均值
print(df)
