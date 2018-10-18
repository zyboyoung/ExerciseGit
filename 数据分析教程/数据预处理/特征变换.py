import numpy as np
import pandas as pd


lst = [6, 8, 10, 15, 16, 24, 25, 40, 67]
print(pd.qcut(lst, q=3, labels=['low', 'medium', 'high']))		# 等频分箱(每组个数相等)

print(pd.cut(lst, bins=3,  labels=['low', 'medium', 'high']))		# 等距分箱(每组间距相等)
