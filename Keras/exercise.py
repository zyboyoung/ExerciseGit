import numpy as np
from keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers import SimpleRNN, LSTM, TimeDistributed, Input

# rnn_lstm中的参数
# 使用含有lstm的rnn做回归，time_steps表示序列步长，input_size表示一个序列中含有一个数据，output_size表示一个序列对应的输出也是一个值
BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


# 使用cnn. rnn作分类时的数据集
def create_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# 这里的维度（-1, 1, 28, 28）分别表示样本数，channel（黑白照片，因此为1，否则为3），长和宽
	x_train = x_train.reshape(-1, 1, 28, 28)
	x_test = x_test.reshape(-1, 1, 28, 28)
	y_train = np_utils.to_categorical(y_train, num_classes=10)
	y_test = np_utils.to_categorical(y_test, num_classes=10)

	return x_train, x_test, y_train, y_test


def regress_fun():
	def create_data():
		X = np.linspace(-1, 1, 200)
		np.random.shuffle(X)
		Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

		plt.scatter(X, Y)
		plt.show()

		X_train, Y_train = X[:160], Y[:160]
		X_test, Y_test = X[160:], Y[160:]

		return X_train, Y_train, X_test, Y_test

	x_train, y_train, x_test, y_test = create_data()

	# 定义模型
	model = Sequential()
	# Dense()中的units参数表示该层的输出维度
	model.add(Dense(units=1, input_dim=1))
	# 添加参数，编译模型
	model.compile(loss='mse', optimizer='sgd')

	# 训练模型
	print('-------- Training --------')
	for step in range(301):
		cost = model.train_on_batch(x_train, y_train)
		if step % 100 == 0:
			print('train cost: ', cost)

	# 检验模型
	print('\n-------- Testing --------')
	cost = model.evaluate(x_test, y_test, batch_size=40)
	print('test cost: ', cost)
	W, b = model.layers[0].get_weights()
	print('Weights=', W, '\nbiases=', b)

	# 可视化
	y_pred = model.predict(x_test)
	plt.scatter(x_test, y_test)
	plt.plot(x_test, y_pred)
	plt.show()


def classifier():
	def create_data():
		# keras自带的可以下载公共的mnist数据集，其中X表示图片，维度为（60000, 28, 28），每个样本的维度为(28, 28)，其中的每个数值范围为（0，255），即像素的范围
		# 而Y表示图片对应的数字，维度为（60000， ），每个样本范围为（0，10）
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		# 对数据集X进行预处理，将(60000, 28, 28)维度转化为(60000, 28*28)，并且通过/255对每个数值进行标准化
		x_train = x_train.reshape(x_train.shape[0], -1) / 255
		x_test = x_test.reshape(x_test.shape[0], -1) / 255

		# 对数据集Y进行预处理，使用keras自带的方法进行one-hot编码
		y_train = np_utils.to_categorical(y_train, num_classes=10)
		y_test = np_utils.to_categorical(y_test, num_classes=10)

		return x_train, x_test, y_train, y_test

	def create_model_1():
		# 第一种方式建立模型，直接在model实例化过程中添加网络层
		model = Sequential([
			Dense(32, input_dim=28*28),
			Activation('relu'),
			Dense(10),
			Activation('softmax'),
		])

		return model

	# def create_model_2():
	# 	# 第二种方式建立模型，即使用model.add
	# 	model = Sequential()
	# 	model.add(Dense(32, input_dim=28*28))
	# 	model.add(Activation('relu'))
	# 	model.add(Dense(10))
	# 	model.add(Activation('softmax'))
	#
	# 	return model

	model = create_model_1()

	x_train, x_test, y_train, y_test = create_data()
	rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
	# 编译模型
	model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
	# 训练模型，epochs表示训练整个训练集的次数，batch_size表示批处理的样本个数
	print('-------- Training --------')
	model.fit(x_train, y_train, epochs=2, batch_size=32)
	# 测试模型
	print('\n-------- Testing --------')
	loss, accuracy = model.evaluate(x_test, y_test)
	print('test loss: ', loss)
	print('test accuracy: ', accuracy)


def create_cnn():
	x_train, x_test, y_train, y_test = create_data()
	model = Sequential()

	# 添加第一个卷积层，(60000, 1, 28, 28) → (60000, 32, 28, 28)
	model.add(Convolution2D(
			filters=32,
			kernel_size=5,
			strides=1,
			padding='same',
			input_shape=(1, 28, 28),
			data_format='channels_first',
	))
	model.add(Activation('relu'))

	# 添加最大池化层，(60000, 32, 28, 28) → (60000, 32, 14, 14), data_format用于指定数据中channel的位置channels_first, channels_last
	model.add(MaxPooling2D(
		pool_size=2,
		strides=2,
		padding='same',
		data_format='channels_first',
	))

	# 添加第二个卷积层，(60000, 32, 14, 14) → (60000, 64, 14, 14)
	model.add(Convolution2D(filters=64, kernel_size=5, strides=5, padding='same'))
	model.add(Activation('relu'))

	# 添加第二个最大池化层，(60000, 64, 14, 14) → (60000, 64, 7, 7)
	model.add(MaxPooling2D(2, 2, padding='same', data_format='channels_first'))

	# 添加第一个全连接层，1024个神经元，需要先压平
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))

	# 添加第二个全连接层，即输出层，10个值
	model.add(Dense(10))

	model.add(Activation('softmax'))

	# 编译模型
	adam = Adam(lr=1e-4)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

	# 训练模型
	print('-------- Training --------')
	model.fit(x_train, y_train, epochs=1, batch_size=256)

	# 测试模型
	print('\n-------- Testing --------')
	loss, accuracy = model.evaluate(x_test, y_test)
	print('test loss: ', loss)
	print('test accuracy: ', accuracy)


def create_rnn():
	# 由于mnist中的图片分辨率是28*28，将其理解为序列化数据，每一行作为一个输入单元input_size(28 p)，一共28行time_steps
	TIME_STEPS = 28
	INPUT_SIZE = 28
	BATCH_SIZE = 50
	BATCH_INDEX = 0
	OUTPUT_SIZE = 10
	CELL_SIZE = 50
	LR = 0.001

	x_train, x_test, y_train, y_test = create_data()

	model = Sequential()

	# 建立模型
	model.add(SimpleRNN(
		batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
		units=CELL_SIZE,
		unroll=True,
	))
	model.add(Dense(OUTPUT_SIZE))
	model.add(Activation('softmax'))

	# 编译模型
	adam = Adam(LR)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

	# 训练模型
	for step in range(4001):
		x_batch = x_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
		y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
		_ = model.train_on_batch(x_batch, y_batch)
		BATCH_INDEX = BATCH_INDEX + BATCH_SIZE
		BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX
		if step % 500 == 0:
			cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
			print('test cost: ', cost)
			print('test accuracy: ', accuracy)


def create_rnn_lstm():
	# 用于生成序列sin(x)，其对应的输出是cos(x)
	def get_batch():
		global BATCH_START, TIME_STEPS
		xs = np.arange(BATCH_START, BATCH_START + BATCH_SIZE * TIME_STEPS).reshape((BATCH_SIZE, TIME_STEPS)) / np.pi
		seq = np.sin(xs)
		res = np.cos(xs)
		BATCH_START += TIME_STEPS
		# plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
		# plt.show()
		# exit(1)
		return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

	model = Sequential()

	# 创建LSTM rnn
	model.add(LSTM(
		batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
		units=CELL_SIZE,
		# 对于每个时间点都会输出值，如果赋值false，则只输出最后的值
		# True: output at all steps. False: output as last step
		return_sequences=True,
		# True: the final state of batch1 is feed into the initial state of batch2
		stateful=True,
	))

	# 添加输出层
	model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

	adam = Adam(lr=LR)
	model.compile(optimizer=adam, loss='mse')

	for step in range(501):
		x_batch, y_batch, xs = get_batch()
		cost = model.train_on_batch(x_batch, y_batch)
		pred = model.predict(x_batch, batch_size=BATCH_SIZE)
		plt.plot(xs[0, :], y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
		plt.ylim((-1.2, 1.2))
		plt.draw()
		plt.pause(0.1)
		if step % 10 == 0:
			print('train cost: ', cost)


def create_autoencorder():
	(x_train, _), (x_test, y_test) = mnist.load_data()

	# 将数据标准化到(-0.5, 0.5)之间，并且x_train维度为（60000，28*28），x_test维度为（10000， 28*28）
	x_train = x_train.astype('float32') / 255. - 0.5
	x_test = x_test.astype('float32') / 255. - 0.5
	x_train = x_train.reshape((x_train.shape[0], -1))
	x_test = x_test.reshape((x_test.shape[0], -1))

	# 设置要压缩成的维度
	encoding_dim = 2

	# 设置input的placeholder
	input_img = Input(shape=(784, ))

	# encoder layers，每层逐步压缩
	encoded = Dense(128, activation='relu')(input_img)
	encoded = Dense(64, activation='relu')(encoded)
	encoded = Dense(10, activation='relu')(encoded)
	encoder_output = Dense(encoding_dim)(encoded)

	# decoder layers
	decoded = Dense(10, activation='relu')(encoder_output)
	decoded = Dense(64, activation='relu')(decoded)
	decoded = Dense(128, activation='relu')(decoded)
	decoded = Dense(784, activation='tanh')(decoded)

	# 组建autoencoder
	autoencorder = Model(inputs=input_img, outputs=decoded)

	# 组建encoder模型，其实就是autoencoder的前半部分
	encoder = Model(inputs=input_img, outputs=encoder_output)

	autoencorder.compile(optimizer='adam', loss='mse')
	autoencorder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)

	# 作图，因为encoder的输出是二维的，因此可以2D的图上显示，颜色则根据分类表示
	encoded_imgs = encoder.predict(x_test)
	plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
	plt.colorbar()
	plt.show()


def create_save():
	X = np.linspace(-1, 1, 200)
	np.random.shuffle(X)
	Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

	x_train, y_train = X[:160], Y[:160]
	x_test, y_test = X[160:], Y[160:]

	model = Sequential()
	model.add(Dense(units=1, input_dim=1))
	model.compile(loss='mse', optimizer='sgd')

	for step in range(301):
		_ = model.train_on_batch(x_train, y_train)

	# 保存模型，格式是h5，由h5py库支持
	print('test before save: ', model.predict(x_test[0:2]))
	model.save('my_model.h5')
	del model

	# load模型
	model = load_model('my_model.h5')
	print('test after save: ', model.predict(x_test[0:2]))

	# 只保存模型中的权重而不保存模型结构
	# model.save_weights('my_model_weights.h5')
	# model.load_weights('my_model_weights.h5')

	# 保存模型的结构
	# from keras.models import model_from_json
	# json_string = model.to_json()
	# model = model_from_json(json_string)


if __name__ == '__main__':
	create_save()
