import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set_context(font_scale=1.5)


# df = pd.read_csv('data/HR.csv', engine='python')
# df = df.dropna(axis=0, how='any')
# df = df[df['last_evaluation'] <= 1][df['salary'] != 'nme'][df['department'] != 'sale']

# sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap=sns.color_palette('RdBu', n_colors=128))
# plt.show()

s1 = pd.Series(['X1', 'X1', 'X2', 'X2', 'X2', 'X2'])
s2 = pd.Series(['Y1', 'Y1', 'Y1', 'Y2', 'Y2', 'Y2'])


def get_entropy(s):
	if not isinstance(s, pd.core.series.Series):
		s = pd.Series(s)
	prt_ary = s.groupby(by=s).count().values / float(len(s))
	return -(np.log2(prt_ary) * prt_ary).sum()

# print('Entropy:', get_entropy(s1))


def get_cond_entropy(s1, s2):
	d = dict()
	for i in list(range(len(s1))):
		d[s1[i]] = d.get(s1[i], []) + [s2[i]]
	return sum([get_entropy(d[k]) * len(d[k]) / float(len(s1)) for k in d])

# print(get_cond_entropy(s1, s2))


def get_entropy_gain(s1, s2):
	return get_entropy(s2) - get_cond_entropy(s1, s2)

# print(get_entropy_gain(s1, s2))


def get_entropy_gain_ratio(s1, s2):
	return get_entropy_gain(s1, s2) / get_entropy(s2)

# print(get_entropy_gain_ratio(s2, s1))


def get_discrete_corr(s1, s2):
	return get_entropy_gain(s1, s2) / math.sqrt(get_entropy(s1) * get_entropy(s2))

# print(get_discrete_corr(s2, s1))


def get_prob_ss(s):
	if not isinstance(s, pd.core.series.Series):
		s = pd.Series(s)
	prt_ary = s.groupby(by=s).count().values / float(len(s))
	return sum(prt_ary ** 2)


def get_gini(s1, s2):
	d = dict()
	for i in list(range(len(s1))):
		d[s1[i]] = d.get(s1[i], []) + [s2[i]]
	return 1 - sum([get_prob_ss(d[k]) * len(d[k]) / float(len(s1)) for k in d])

print(get_gini(s2, s1))
