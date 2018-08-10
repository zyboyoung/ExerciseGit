from KerasFramework.dataset import normalize_vectors
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from time import time


def create_model(input_shape):
	X_input = Input(input_shape)

	X = ZeroPadding2D((3, 3))(X_input)

	X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
	X = BatchNormalization(axis=3, name='bn0')(X)
	X = Activation('relu')(X)

	X = MaxPooling2D((2, 2), name='max_pool')(X)

	X = Flatten()(X)
	X = Dense(1, activation='sigmoid', name='fc')(X)

	model = Model(inputs=X_input, outputs=X, name='HappyModel')

	return model


if __name__ == '__main__':
	start_time = time()

	X_train, X_test, Y_train, Y_test, classes = normalize_vectors()

	# X_train: (600, 64, 64, 3), X_train.shape[1:]: (64, 64, 3)
	happy_model = create_model(X_train.shape[1:])
	happy_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
	happy_model.fit(x=X_train, y=Y_train, epochs=100, batch_size=64)

	preds = happy_model.evaluate(x=X_test, y=Y_test)

	print('Loss = ' + str(preds[0]))
	print('Test Accuracy = ' + str(preds[1]))

	end_time = time()
	print(end_time - start_time, ' s')
