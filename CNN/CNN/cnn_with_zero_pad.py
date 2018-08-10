import numpy as np
from time import time


def zero_pad(x, pad):
	"""
	Pad all images of the dataset X with zeros
	:param x: (m, n_H, n_W, n_C)
	:param pad: integer
	:return: x_pad, padded image (m, n_H+2*pad, n_W+2*pad, n_C)
	"""
	# x第一维是样本量，第四维是信号通道，只需要对height和width做padding
	x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

	return x_pad


def conv_single_step(a_slice_prev, W, b):
	"""
	:param a_slice_prev: 将输入划分为与w一致的shape (f, f, n_C_prev)
	:param W: (f, f, n_C_prev)
	:param b: matrix of shape (1, 1, 1)
	:return: Z-一个具体的值
	"""
	s = np.multiply(a_slice_prev, W) + b
	Z = np.sum(s)

	return Z


def conv_forward(A_prev, W, b, hparameters):
	"""
	Forward propagation for a convolution function
	:param A_prev: 前一层卷积后并激活后的输出，(m, n_H_prev, n_W_prev, n_C_prev)
	:param W: (f, f, n_C_prev, n_C)
	:param b: (1, 1, 1, n_C)
	:param hparameters: stride and pad
	:return: Z-(m, n_H, n_W, n_C), cache
	"""
	# 将各种维度、各种参数提取出来
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	(f, f, n_C_prev, n_C) = W.shape()
	stride = hparameters['stride']
	pad = hparameters['pad']

	# 计算conv层输出的维度
	n_H = int((n_H_prev + 2 * pad - f) / stride + 1)
	n_W = int((n_W_prev + 2 * pad - f) / stride + 1)

	# 初始化输出
	Z = np.zeros((m, n_H, n_W, n_C))

	# 利用pad、A_prev创建A_prev_pad
	A_prev_pad = zero_pad(A_prev, pad)

	for i in range(m):
		# 从m个样本中提取每个样本a_prev_pad
		a_prev_pad = A_prev_pad[i, :, :, :]

		for h in range(0, n_H, stride):
			for W in range(0, n_W, stride):
				for c in range(0, n_C):
					# 分别定位到每次切片的竖直方向的起点、结尾，水平方向的起点、结尾
					vert_start = h
					vert_end = h + f
					horiz_start = W
					horiz_end = W + f

					# 进一步定位到每个切片
					a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

					# 进行一步卷积操作
					Z[i, h, W, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

	assert (Z.shape == (m, n_H, n_W, n_C))
	cache = (A_prev, W, b, hparameters)

	return Z, cache


def pool_forward(A_prev, hparameters, mode='max'):
	"""
	:param A_prev: (m, n_H_prev, n_W_prev, n_C_prev)
	:param hparameters: f and stride
	:param mode: 'max' or 'average'
	:return: A-(m, n_H, n_W, n_C), cache
	"""
	# 将各种维度、各种参数提取出来
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	stride = hparameters['stride']
	f = hparameters['f']

	# 计算池化后的维度, 池化不改变信号通道数
	n_H = int((n_H_prev - f) / stride + 1)
	n_W = int((n_W_prev - f) / stride + 1)
	n_C = n_C_prev

	A = np.zeros((m, n_H, n_W, n_C))
	for i in range(m):
		for h in range(0, n_H, stride):
			for w in range(0, n_W, stride):
				for c in range(0, n_C):
					vert_start = h
					vert_end = h + f
					horiz_start = w
					horiz_end = w + f

					a_prev_slice = A_prev[i, vert_start: vert_end, horiz_start: horiz_end, c]

					if mode == 'max':
						A[i, h, w, c] = np.max(a_prev_slice)
					elif mode == 'average':
						A[i, h, w, c] = np.mean(a_prev_slice)

	assert (A.shape == (m, n_H, n_W, n_C))

	cache = (A_prev, hparameters)

	return A, cache


def conv_backward(dZ, cache):
	"""
	Backward propagation for a convolution function
	:param dZ: (m, n_H, n_W, n_C)
	:param cache:
	:return: dA_prev-(m, n_H_prev, n_W_prev, n_C_prev), dW-(f, f, n_C_prev, n_C), db-(1, 1, 1, n_C)
	"""
	(A_prev, W, b, hparameters) = cache
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	(f, f, n_C_prev, n_C) = W.shape
	stride = hparameters['stride']
	pad = hparameters['pad']
	(m, n_H, n_W, n_C) = dZ.shape

	dA_prev = np.zeros(A_prev.shape)
	dW = np.zeros(W.shape)
	db = np.zeros(b.shape)

	A_prev_pad = zero_pad(A_prev, pad)
	dA_prev_pad = zero_pad(dA_prev, pad)

	for i in range(m):
		a_prev_pad = A_prev_pad[i, :, :, :]
		da_prev_pad = dA_prev_pad[i, :, :, :]

		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):
					vert_start = h * stride
					vert_end = vert_start + f
					horiz_start = w * stride
					horiz_end = horiz_start + f

					a_slice = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]
					da_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
					dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
					db[:, :, :, c] += dZ[i, h, w, c]

		dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

	assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

	return dA_prev, dW, db


def create_mask_from_window(x):
	"""
	Creates a mask from an input matrix x. 在最大值位置取值1，其它位置取值0，用于帮助最大池化层的反向传播.
	对于最大池化，反向传播中的梯度应该反映到最大值所在的位置上
	:param x: (f, f)
	:return: mask-(f, f)
	"""
	mask = (x == np.max(x))

	return mask


def distribute_value(dz, shape):
	"""
	对于平均池化，反向传播中的梯度应该反映到每一个位置上
	:param dz: 梯度值
	:param shape: (n_H, n_W)
	:return: a-将dz的值平均分配到(n_H, n_W)上
	"""
	(n_H, n_W) = shape
	average = dz / (n_H * n_W)
	a = np.ones(shape) * average

	return a


def pool_backward(dA, cache, mode='max'):
	"""
	Back propagation for pooling layer
	:param dA:
	:param cache:
	:param mode: 'max' or 'average'
 	:return: dA_prev
 	"""
	(A_prev, hparameters) = cache
	stride = hparameters['stride']
	f = hparameters['f']
	(m, n_H, n_W, n_C) = dA.shape

	dA_prev = np.zeros(A_prev.shape)

	for i in range(m):
		a_prev = A_prev[i, :, :, :]

		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):
					vert_start = h * stride
					vert_end = vert_start + f
					horiz_start = w * stride
					horiz_end = horiz_start + f

					if mode == 'max':
						a_prev_slice = a_prev[vert_start: vert_end, horiz_start: horiz_end, c]
						mask = create_mask_from_window(a_prev_slice)
						dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
					elif mode == 'average':
						da = dA[i, h, w, c]
						shape = (f, f)
						dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

	assert (dA_prev.shape == A_prev.shape)

	return dA_prev


if __name__ == '__main__':
	start_time = time()

	end_time = time()

	print('------cost time: ' + str(end_time - start_time) + ' s ------')
