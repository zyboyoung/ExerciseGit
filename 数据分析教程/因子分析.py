import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.decomposition import PCA
sns.set_context(font_scale=1.5)
my_pca = PCA(n_components=7)


df = pd.read_csv('data/HR.csv', engine='python')
df = df.dropna(axis=0, how='any')
df = df[df['last_evaluation'] <= 1][df['salary'] != 'nme'][df['department'] != 'sale']

lower_mat = my_pca.fit_transform(df.drop(labels=['salary', 'department', 'left'], axis=1))
print('ratio:', my_pca.explained_variance_ratio_)
sns.heatmap(pd.DataFrame(lower_mat).corr(), vmin=-1, vmax=1, cmap=sns.color_palette('RdBu', n_colors=128))
plt.show()
