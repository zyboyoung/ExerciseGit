import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, f1_score, recall_score


iris = datasets.load_iris()
labels = np.copy(iris.target)		# 一共有三类，取值为0，1，2
random_unlabeled_points = np.random.rand(len(iris.target))
random_unlabeled_points = random_unlabeled_points < 0.7
y = labels[random_unlabeled_points]			# 即将被修改为-1的原标注
labels[random_unlabeled_points] = -1		# 将原来的部分标注改为了-1，成为了半监督学习
print('Unlabeled Number: ', list(labels).count(-1))

label_prop_model = LabelPropagation()
label_prop_model.fit(iris.data, labels)
y_pred = label_prop_model.predict(iris.data)
y_pred = y_pred[random_unlabeled_points]	# 只选择没有标注的样本的预测结果
print('ACC: ', accuracy_score(y, y_pred))
print('REC: ', recall_score(y, y_pred, average='micro'))
print('F1: ', f1_score(y, y_pred, average='micro'))
