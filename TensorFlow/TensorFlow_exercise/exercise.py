import tensorflow as tf
import numpy as np
import math


def linear_function():
	"""
	创建线性函数，初始化X、W、b
	:return:
	"""
	# 初始化
	X = tf.constant(np.random.randn(3, 1), name='X')
	W = tf.constant(np.random.randn(4, 3), name='W')
	b = tf.constant(np.random.randn(4, 1), name='b')

	# 定义op
	Y = tf.add(tf.matmul(W, X), b)

	# 创建session并运行
	session = tf.Session()
	result = session.run(Y)
	session.close()

	return result


def sigmoid(z):
	"""
	创建sigmoid函数
	:param z: input data
	:return:
	"""
	x = tf.placeholder(tf.float32, name='x')

	sigmoid = tf.sigmoid(x)

	with tf.Session() as session:
		result = session.run(sigmoid, feed_dict={x:z})

	return result


def cost(logits, labels):
	"""
	计算cost
	:param logits: Z[L]
	:param labels: y(1 or 0)
	:return:
	"""
	z = tf.placeholder(tf.float32, name='z')
	y = tf.placeholder(tf.float32, name='y')

	cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

	with tf.Session() as session:
		result = session.run(cost, feed_dict={z:logits, y:labels})

	return result


def one_hot_matrix(labels, C):
	"""
	将原本的labels向量转化为元素全为0或1的矩阵，第i行对应第i个类别，第j列对应第j个样本
	:param labels:
	:param C: 类别数量
	:return:
	"""
	C = tf.constant(C, name='C')
	one_hot_matrix = tf.one_hot(labels, depth=C, axis=0)

	with tf.Session() as session:
		one_hot = session.run(one_hot_matrix)

	return one_hot


def ones(shape):
	ones = tf.ones(shape)
	with tf.Session() as session:
		ones = session.run(ones)

	return ones


def random_mini_batches(X, Y, mini_batch_size=64):
	"""
	需要先将原始数据进行随机分散，再按照mini_batch_size进行划分
	:param X: input
	:param Y: label
	:param mini_batch_size: 大小一般为2的幂
	:return: mini_batches: [(mini_batch_X_1, mini_batch_Y_1), (mini_batch_X_2, mini_batch_Y_2), ...]
	"""

	m = X.shape[1]
	mini_batches = []

	# shuffle
	shuffled_m = list(np.random.permutation(m))
	shuffled_X = X[:, shuffled_m]
	shuffled_Y = Y[:, shuffled_m]

	# partition
	# 向下取整获得完整的batches数
	num_complete_batches = math.floor(m / mini_batch_size)
	for i in range(num_complete_batches):
		mini_batch_X = shuffled_X[:, i * mini_batch_size: (i+1) * mini_batch_size]
		mini_batch_Y = shuffled_Y[:, i * mini_batch_size: (i+1) * mini_batch_size]

		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	# 处理剩余部分数据
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_batches* mini_batch_size: m]
		mini_batch_Y = shuffled_Y[:, num_complete_batches* mini_batch_size: m]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches


def convert_to_one_hot(Y, C):
	Y = np.eye(C)[Y.reshape(-1)].T
	return Y


# if __name__=='__main__':
	# print(linear_function())

	# print(sigmoid(12))

	# logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
	# cost = cost(logits, np.array([0, 0, 1, 1]))
	# print('logits= '+str(logits))
	# print('cost= '+str(cost))

	# labels = np.array([1, 2, 3, 0, 2, 1])
	# one_hot = one_hot_matrix(labels, C=4)
	# print('one_hot = '+str(one_hot))

	# print('ones = '+str(ones([3])))