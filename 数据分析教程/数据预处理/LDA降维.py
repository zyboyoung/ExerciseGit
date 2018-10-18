import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

print(LinearDiscriminantAnalysis(n_components=1).fit_transform(x, y))		# 使用LDA降维，可以指定降到的维度

clf = LinearDiscriminantAnalysis(n_components=1).fit(x, y)		# 也可以用于分类
print(clf.predict([[0.8, 1]]))
