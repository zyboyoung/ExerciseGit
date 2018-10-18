import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel


df = pd.DataFrame({'A': ss.norm.rvs(size=10), 'B': ss.norm.rvs(size=10), 'C': ss.norm.rvs(size=10), 'D': np.random.randint(low=0, high=2, size=10)})
print(df)

x = df.loc[:, ['A', 'B', 'C']]
y = df.loc[:, 'D']

skb = SelectKBest(k=2)		# 采用过滤思想，可以指定方式，默认是采用F-检验(方差检验)
skb.fit(x, y)
print(skb.transform(x))

rfe = RFE(estimator=SVR(kernel='linear'), n_features_to_select=2, step=1)		# 采用包裹思想，指定回归模型，选择后剩余的特征数，以及每次迭代去除的特征量
print(rfe.fit_transform(x, y))

sfm = SelectFromModel(estimator=DecisionTreeRegressor(), threshold=0.1)		# 采用嵌入思想，threshold表示特征权重低于多少时就被去除
print(sfm.fit_transform(x, y))
