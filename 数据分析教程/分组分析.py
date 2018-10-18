import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(font_scale=1.5)


df = pd.read_csv('data/HR.csv', engine='python')
df = df.dropna(axis=0, how='any')
df = df[df['last_evaluation'] <= 1][df['salary'] != 'nme'][df['department'] != 'sale']

# sns.barplot(x='salary', y='left', hue='department', data=df)
# plt.show()

sl_s = df['satisfaction_level']
sns.barplot(list(range(len(sl_s))), sl_s.sort_values())
plt.show()
