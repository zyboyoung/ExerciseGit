import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from time import time


def train_preprocess(filename):
	data = pd.read_csv(filename)
	data = data.fillna(0)
	data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

	data_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
	data_X = data_X.values


	data['Deceased'] = data['Survived'].apply(lambda x: int(not x))
	data_Y = data[['Deceased', 'Survived']]
	data_Y = data_Y.values

	return data_X, data_Y


def test_preprocess(filename):
	data = pd.read_csv(filename)
	data = data.fillna(0)
	data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

	data_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
	data_X = data_X.values

	return data_X


def tf_process():
	# 声明输入占位符
	X = tf.placeholder(tf.float32, shape=[None, 6])
	y = tf.placeholder(tf.float32, shape=[None, 2])

	# 声明参数变量
	weights = tf.Variable(tf.random_normal([6, 2]), name='weights')
	bias = tf.Variable(tf.zeros([2]), name='bias')

	# 构造向前传播计算图
	y_pred = tf.nn.softmax(tf.matmul(X, weights) + bias)

	# 声明代价函数
	cross_entropy = - tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
	cost = tf.reduce_mean(cross_entropy)

	# 声明训练算子，并加入优化算法
	train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

	# 声明在验证集上计算精度的算子
	correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
	accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# 创建、运行session
	with tf.Session() as session:
		tf.global_variables_initializer().run()
		for epoch in range(1000):
			total_loss = 0.
			for i in range(len(X_train)):
				_, loss = session.run([train_op, cost], feed_dict={X: [X_train[i]], y: [y_train[i]]})
				total_loss += loss
			print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))
		print('Training complete!')

		accuracy = session.run(accuracy_op, feed_dict={X: X_val, y: y_val})
		print('Accuracy on validation set: %.9f' % accuracy)

		prediction = np.argmax(session.run(y_pred, feed_dict={X: X_test}), 1)
		submission = pd.DataFrame({
			'PassengerId': test_data['PassengerId'],
			'Survived': prediction
		})

	return submission


if __name__ == '__main__':
	start_time = time()

	raw_data_X, raw_data_Y = train_preprocess('train.csv')
	X_train, X_val, y_train, y_val = train_test_split(raw_data_X, raw_data_Y, test_size=0.2, random_state=42)

	X_test = test_preprocess('test.csv')
	test_data = pd.read_csv('test.csv')

	print(tf_process())

	end_time = time()
	print(end_time - start_time, 's')
