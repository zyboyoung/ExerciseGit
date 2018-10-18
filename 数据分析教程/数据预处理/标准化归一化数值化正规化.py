import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, Normalizer


print(MinMaxScaler().fit_transform(np.array([1.0, 4.0, 10.0, 15.0, 21.0]).reshape(-1, 1)))		# 归一化

print(StandardScaler().fit_transform(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)))		# 标准化

print(LabelEncoder().fit_transform(np.array(['Down', 'Up', 'Down', 'Up']).reshape(-1, 1)))		# 标签编码

lb_encoder = LabelEncoder()
lb_trans = lb_encoder.fit_transform(np.array(['Red', 'Yellow', 'Blue', 'Green']))
one_hot_encoder = OneHotEncoder().fit(lb_trans.reshape(-1, 1))
print(one_hot_encoder.transform(lb_encoder.transform(np.array(['Yellow', 'Blue', 'Green', 'Green', 'Red'])).reshape(-1, 1)).toarray())		# one_hot编码

print(Normalizer(norm='l1').fit_transform(np.array([[1.0, 1.0, 3.0, -1.0, 2.0]])))			# L1正规化，正规化是对特征行进行的
print(Normalizer(norm='l2').fit_transform(np.array([[1.0, 1.0, 3.0, -1.0, 2.0]])))			# L2正规化，正规化是对特征行进行的
