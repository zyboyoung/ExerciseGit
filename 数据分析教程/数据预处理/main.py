import pandas as pd
import numpy as np
import os
import pydotplus
from functools import wraps
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_squared_error
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB, BernoulliNB		# 高斯朴素贝叶斯可以用于连续值、离散值，伯努利朴素贝叶斯用于0，1值
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

pd.set_option('max_columns', None)
os.environ['PATH'] += os.pathsep + 'D:/Tools/GraphViz/release/bin/'


def caltime(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print('-' * 8)
		print('start time: ', time.asctime(time.localtime(start)))
		print('end time:   ', time.asctime(time.localtime(end)))
		print('-' * 8)
		cost_time = end - start
		if cost_time < 1:
			print(func.__name__, '{:.5f}'.format(cost_time * 1000), 'ms')
		else:
			cost_time = '{:.2f}'.format(cost_time)
			print(func.__name__, cost_time, 's')
		print('-' * 8)
		return result

	return wrapper


def hr_preprocessing(sl=False,
					 le=False,
					 npr=False,
					 amh=False,
					 tsc=False,
					 wa=False,
					 pl5=False,
					 dp=False,
					 slr=False,
					 lower_d=False,
					 ld_n=1):
	# sl: satisfaction_level---False: MinMaxScaler; True: StandardScaler
	# le: last_evaluation---False: MinMaxScaler; True: StandardScaler
	# npr: number_project---False: MinMaxScaler; True: StandardScaler
	# amh: average_monthly_hours---False: MinMaxScaler; True: StandardScaler
	# tsc: time_spend_company---False: MinMaxScaler; True: StandardScaler
	# wa: Work_accident---False: MinMaxScaler; True: StandardScaler
	# pl5: promotion_last_5years---False: MinMaxScaler; True: StandardScaler
	# dp: department---False: LabelEncoding; True: OneHotEncoding
	# slr: salary---False: LabelEncoding; True: OneHotEncoding

	df = pd.read_csv('../data/HR.csv')

	# 1. 清洗数据
	df = df.dropna(subset=['satisfaction_level', 'last_evaluation'])
	df = df[df['satisfaction_level'] <= 1][df['salary'] != 'nme']

	# 2. 得到标注
	label = df['left']
	df = df.drop('left', axis=1)		# axis=1表示指定列，默认axis=0指定行

	# 3. 特征选择：可以根据相关分析中找出和label不相关的特征，进行去除
	# 4. 特征处理
	scaler_1 = [sl, le, npr, amh, tsc, wa, pl5]
	column_1 = ['satisfaction_level',
				'last_evaluation',
				'number_project',
				'average_monthly_hours',
				'time_spend_company',
				'Work_accident',
				'promotion_last_5years']
	for i in range(len(scaler_1)):
		if not scaler_1[i]:
			df[column_1[i]] = MinMaxScaler().fit_transform(df[column_1[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
		else:
			df[column_1[i]] = StandardScaler().fit_transform(df[column_1[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

	scaler_2 = [dp, slr]
	column_2 = ['department', 'salary']
	for i in range(len(scaler_2)):
		if not scaler_2[i]:
			if column_2[i] == 'salary':
				df[column_2[i]] = [map_salary(s) for s in df['salary'].values]
			else:
				df[column_2[i]] = LabelEncoder().fit_transform(df[column_2[i]])
			df[column_2[i]] = StandardScaler().fit_transform(df[column_2[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
		else:
			df = pd.get_dummies(df, columns=[column_2[i]])

	# 5. 降维处理
	if lower_d:
		return PCA(n_components=ld_n).fit_transform(df.values), label

	return df, label


def map_salary(s):
	d = dict([('low', 0), ('medium', 1), ('high', 2)])
	return d.get(s, 0)


def hr_modeling(features, label, tree_vis=False, ann=False):
	# 将原数据集划分为了训练集、验证集、测试集，6：2：2切分
	f_v = features.values
	f_n = features.columns.values
	l_v = label.values
	x_tt, x_validation, y_tt, y_validation = train_test_split(f_v, l_v, test_size=0.2)
	x_train, x_test, y_train, y_test = train_test_split(x_tt, y_tt, test_size=0.25)

	models = []
	models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
	models.append(('GaussianNB', GaussianNB()))
	models.append(('BernoulliNB', BernoulliNB()))
	models.append(('DecisionTreeGini', DecisionTreeClassifier()))
	models.append(('DecisionTreeEntropy', DecisionTreeClassifier(criterion='entropy')))
	models.append(('SVM Classifier', SVC(C=1000)))
	models.append(('RandomForest', RandomForestClassifier(n_estimators=81, max_features=None)))
	models.append(('Adaboost', AdaBoostClassifier()))
	models.append(('LogisticRegression', LogisticRegression(penalty='l2', C=1.0, tol=1e-10)))
	models.append(('GBDT', GradientBoostingClassifier(max_depth=6, n_estimators=100)))
	for clf_name, clf in models:
		clf.fit(x_train, y_train)
		xy_list = [(x_train, y_train), (x_validation, y_validation), (x_test, y_test)]
		for i in range(len(xy_list)):
			x_part = xy_list[i][0]
			y_part = xy_list[i][1]
			y_pred = clf.predict(x_part)
			print(i)		# 分别将模型在训练集、验证集、测试集上进行实验
			print(clf_name, '-ACC:', accuracy_score(y_part, y_pred))
			print(clf_name, '-REC:', recall_score(y_part, y_pred))
			print(clf_name, '-F1:', f1_score(y_part, y_pred))

		# 决策树可以生成图
		if clf_name.startswith('DecisionTree') and tree_vis:
			dot_data = export_graphviz(clf, out_file=None,
									   feature_names=f_n,
									   class_names=['NL', 'L'],
									   filled=True, rounded=True,
									   special_characters=True)
			graph = pydotplus.graph_from_dot_data(dot_data)
			graph.write_pdf('dt_tree.pdf')

	if ann:
		ann_model = Sequential()
		ann_model.add(Dense(50, input_dim=len(f_v[0])))
		ann_model.add(Activation('sigmoid'))
		ann_model.add(Dense(2))
		ann_model.add(Activation('softmax'))
		sgd = SGD(lr=0.1)
		ann_model.compile(optimizer=sgd, loss='mean_squared_error')
		ann_model.fit(x=x_train, y=np.array([[0, 1] if i==1 else [1, 0] for i in y_train]), nb_epoch=15000, batch_size=8999)
		xy_list = [(x_train, y_train), (x_validation, y_validation), (x_test, y_test)]
		for i in range(len(xy_list)):
			x_part = xy_list[i][0]
			y_part = xy_list[i][1]
			y_pred = ann_model.predict_classes(x_part)
			print(i)
			print('ANN', '-ACC:', accuracy_score(y_part, y_pred))
			print('ANN', '-REC:', recall_score(y_part, y_pred))
			print('ANN', '-F1:', f1_score(y_part, y_pred))


def regr_test(features, label):
	print('X', features)
	print('Y', label)
	regr = [LinearRegression(), Lasso(alpha=0.002), Ridge(alpha=0.8)]
	for i in range(len(regr)):
		regr[i].fit(features.values, label.values)
		y_pred = regr[i].predict(features.values)
		print('Coef ' + str(i) + ':', regr[i].coef_)		# 线性回归的两个参数
		print('MSE ' + str(i) + ':', mean_squared_error(y_pred, label.values))


@caltime
def main(model=True, regression=False):
	features, label = hr_preprocessing()
	if model:
		hr_modeling(features, label)
	if regression:
		regr_test(features[['number_project', 'average_monthly_hours']], features['last_evaluation'])


if __name__ == '__main__':
	main(model=True, regression=False)
