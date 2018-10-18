import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('data/HR.csv', engine='python')
df = df.dropna(axis=0, how='any')
df = df[df['last_evaluation'] <= 1][df['salary'] != 'nme'][df['department'] != 'sale']

# 使用seaborn画柱状图
sns.set_style(style='whitegrid')
sns.set_context(context='paper', font_scale=1)
sns.set_palette('summer')
# sns.countplot(x='salary', hue='department', data=df)
# plt.show()

# 使用基本的pyplot画出柱状图
# plt.bar(np.arange(len(df['salary'].value_counts())) + 0.5, df['salary'].value_counts(), width=0.5)
# plt.title('SALARY')
# plt.xlabel('salary')
# plt.ylabel('Number')
# plt.xticks(np.arange(len(df['salary'].value_counts())) + 0.5, df['salary'].value_counts().index)
# plt.axis([0, 4, 0, 10000])		# 分别指定的是x轴最小值、最大值，y轴的最小值、最大值
#
# for x,y in zip(np.arange(len(df['salary'].value_counts())) + 0.5, df['salary'].value_counts()):
# 	plt.text(x, y, y, ha='center', va='bottom')		# 为每个点(x,y)处加上标注(y的值)，水平位置是center，垂直位置是底部
#
# plt.show()

# 画出直方图
# f = plt.figure()
# f.add_subplot(1, 3, 1)
# sns.distplot(df['satisfaction_level'], bins=10)
# f.add_subplot(1, 3, 2)
# sns.distplot(df['last_evaluation'], bins=10)
# f.add_subplot(1, 3, 3)
# sns.distplot(df['average_monthly_hours'], bins=10)
# plt.show()

# 箱线图box_plot
# sns.boxplot(x=df['time_spend_company'], saturation=0.75, whis=3)
# plt.show()

# 折线图point_plot
# sub_df = df.groupby('time_spend_company').mean()
# sns.pointplot(sub_df.index, sub_df['left'])
# # sns.pointplot(x=df['time_spend_company'], y=df['left'], data=df)
# plt.show()

# 饼图(seaborn没有饼图，只能使用matplotlib)
explodes = [0.1 if i == 'sales' else 0 for i in df['department'].value_counts().index]
plt.pie(df['department'].value_counts(normalize=True), labels=df['department'].value_counts().index, explode=explodes, autopct='%1.1f%%')
plt.show()
