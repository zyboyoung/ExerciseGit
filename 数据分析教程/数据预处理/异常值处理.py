import numpy as np
import pandas as pd


df = pd.DataFrame({'A': ['a0', 'a1', 'a1', 'a2', 'a3', 'a4'], 'B': ['b0', 'b1', 'b2', 'b2', 'b3', None], 'C': [1, 2, None, 3, 4, 5], 'D': [0.1, 10.2, 11.4, 8.9, 9.1, 12], 'E': [10, 19, 32, 25, 8, None], 'F': ['f0', 'f1', 'g2', 'f3', 'f4', 'f5']})

df.isnull()
df.dropna()		# 去掉的是含有空值的行
df.dropna(subset=[''])	# 指定某列，对其中含有空值的行，整个df删除该行