import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as functional
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torchvision
import time

# torch.unsqueeze(tensor, dim=1)将原tensor维度转化为（-1， 1），dim=1或者dim=-2表示维度转化为（-1， 1），dim=0或者dim=-1表示维度转化为（1， -1）
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x), Variable(y)


def torch_numpy():
	# numpy与torch的数据转化
	np_data = np.arange(6).reshape((2, 3))
	torch_data = torch.from_numpy(np_data)
	tensor2array = torch_data.numpy()

	print(
		'\nnumpy', np_data,
		'\ntorch', torch_data,
		'\ntensor2array', tensor2array,
	)


def torch_variable():
	# 构建一个torch的Variable变量，并进行正向和反向传播
	tensor = torch.FloatTensor([[1, 2], [3, 4]])
	# variable是Variable类型数据，可以通过variable.data获取tensor类型数据
	variable = Variable(tensor, requires_grad=True)

	# 正向传播
	v_out = torch.mean(variable * variable)
	# 反向传播
	v_out.backward()
	print(variable.grad)
	print(variable.data.numpy())


def torch_activation():
	# 构建torch的激活函数，时刻注意，torch中是数据类型是tensor，而在可视化或者后续数据处理中，需要转化为np数据
	# 生成-5到5的200个数
	x = torch.linspace(-5, 5, 200)
	x = Variable(x)
	x_np = x.data.numpy()

	y_relu = functional.relu(x).data.numpy()
	y_sigmoid = functional.sigmoid(x).data.numpy()
	y_tanh = functional.tanh(x).data.numpy()
	y_softplus = functional.softplus(x).data.numpy()
	# y_softmax = functional.softmax(x)

	plt.figure(1, figsize=(8, 6))

	plt.subplot(221)
	plt.plot(x_np, y_relu, c='red', label='relu')
	plt.ylim((-1, 5))
	plt.legend(loc='best')

	plt.subplot(222)
	plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
	plt.ylim((-0.2, 1.2))
	plt.legend(loc='best')

	plt.subplot(223)
	plt.plot(x_np, y_tanh, c='red', label='tanh')
	plt.ylim((-1.2, 1.2))
	plt.legend(loc='best')

	plt.subplot(224)
	plt.plot(x_np, y_softplus, c='red', label='softplus')
	plt.ylim((-0.2, 6))
	plt.legend(loc='best')

	plt.show()


def torch_regression():
	plt.scatter(x.data.numpy(), y.data.numpy())
	plt.show()

	class Net(torch.nn.Module):
		def __init__(self, n_feature, n_hidden, n_output):
			super(Net, self).__init__()
			self.hidden = torch.nn.Linear(n_feature, n_hidden)
			self.predict = torch.nn.Linear(n_hidden, n_output)

		def forward(self, x):
			x = functional.relu(self.hidden(x))
			x = self.predict(x)
			return x

	net = Net(1, 10, 1)
	print(net)

	plt.ion()
	plt.show()

	optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
	loss_func = torch.nn.MSELoss()

	for t in range(100):
		prediction = net(x)

		loss = loss_func(prediction, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if t % 5 == 0:
			plt.cla()
			plt.scatter(x.data.numpy(), y.data.numpy())
			plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
			plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
			plt.pause(0.1)

	plt.ioff()
	plt.show()


def torch_classify():
	# 生成(100, 2)的tensor型数据，全为1
	n_data = torch.ones(100, 2)
	x0 = torch.normal(2 * n_data, 1)
	y0 = torch.zeros(100)
	x1 = torch.normal(-2 * n_data, 1)
	y1 = torch.ones(100)

	x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
	y = torch.cat((y0, y1), 0).type(torch.LongTensor)

	x, y = Variable(x), Variable(y)

	# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
	# plt.show()

	# method1
	class Net(torch.nn.Module):
		def __init__(self, n_feature, n_hidden, n_output):
			super(Net, self).__init__()
			self.hidden = torch.nn.Linear(n_feature, n_hidden)
			self.out = torch.nn.Linear(n_hidden, n_output)

		def forward(self, x):
			x = functional.relu(self.hidden(x))
			x = self.out(x)
			return x

	net = Net(2, 10, 2)
	print(net)

	# method2
	net2 = torch.nn.Sequential(
		torch.nn.Linear(2, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 2),
	)

	print(net2)

	plt.ion()
	plt.show()

	optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
	loss_func = torch.nn.CrossEntropyLoss()

	for t in range(100):
		out = net(x)

		loss = loss_func(out, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if t % 10 == 0 or t in [3, 6]:
			plt.cla()
			prediction = torch.max(functional.softmax(out, dim=1), 1)[1]
			pred_y = prediction.data.numpy().squeeze()
			target_y = y.data.numpy()
			plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
			accuracy = sum(target_y == pred_y) / 200.0
			plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
			plt.pause(0.1)

	plt.ioff()
	plt.show()


def torch_save():
	net = torch.nn.Sequential(
		torch.nn.Linear(1, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1),
	)

	optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
	loss_func = torch.nn.MSELoss()

	for t in range(100):
		prediction = net(x)
		loss = loss_func(prediction, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	torch.save(net, 'net.pkl')
	torch.save(net.state_dict(), 'net_params.pkl')

	plt.figure(1, figsize=(10, 3))
	plt.subplot(131)
	plt.title('Net1')
	plt.scatter(x.data.numpy(), y.data.numpy())
	plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def torch_restore_net():
	net = torch.load('net.pkl')
	prediction = net(x)

	plt.figure(1, figsize=(10, 3))
	plt.subplot(132)
	plt.title('Net2')
	plt.scatter(x.data.numpy(), y.data.numpy())
	plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def torch_restore_params():
	net = torch.nn.Sequential(
		torch.nn.Linear(1, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1),
	)

	net.load_state_dict(torch.load('net_params.pkl'))
	prediction = net(x)

	plt.figure(1, figsize=(10, 3))
	plt.subplot(133)
	plt.title('Net3')
	plt.scatter(x.data.numpy(), y.data.numpy())
	plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
	plt.show()


def torch_batch():
	BATCH_SIZE = 5

	x = torch.linspace(1, 10, 10)
	y = torch.linspace(10, 1, 10)

	raw_dataset = Data.TensorDataset(x, y)
	loader = Data.DataLoader(
		dataset=raw_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2,
	)

	for epoch in range(3):
		for step, (batch_x, batch_y) in enumerate(loader):
			print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch_y: ', batch_y.numpy())


def torch_optimizer():
	LR = 0.01
	BATCH_SIZE = 32
	EPOCH = 12

	x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
	y = x.pow(2) + 0.1 * torch.normal(torch.zeros(x.size()))

	# plt.scatter(x.numpy(), y.numpy())
	# plt.show()

	data = Data.TensorDataset(x, y)
	loader = Data.DataLoader(
		dataset=data,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2,
	)

	class Net(torch.nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.hidden = torch.nn.Linear(1, 20)
			self.predict = torch.nn.Linear(20, 1)

		def forward(self, x):
			x = torch.nn.functional.relu(self.hidden(x))
			x = self.predict(x)
			return x

	net_SGD = Net()
	net_Momentum = Net()
	net_RMSprop = Net()
	net_Adam = Net()
	nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

	opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
	opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
	opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
	opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
	optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

	loss_func = torch.nn.MSELoss()
	losses_his = [[], [], [], []]

	for epoch in range(0, EPOCH):
		print(epoch)
		for step, (batch_x, batch_y) in enumerate(loader):
			b_x = Variable(batch_x)
			b_y = Variable(batch_y)

			for net, opt, l_his in zip(nets, optimizers, losses_his):
				output = net(b_x)
				loss = loss_func(output, b_y)
				opt.zero_grad()
				loss.backward()
				opt.step()
				l_his.append(loss.data)

	labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
	for i, l_his in enumerate(losses_his):
		plt.plot(l_his, label=labels[i])
	plt.legend(loc='best')
	plt.xlabel('Steps')
	plt.ylabel('Loss')
	plt.ylim((0, 0.2))
	plt.show()


def torch_cnn():
	EPOCH = 1
	BATCH_SIZE = 50
	LR = 0.001
	DOWNLOAD_MNIST = False

	train_data = torchvision.datasets.MNIST(
		root='./mnist',
		train=True,
		transform=torchvision.transforms.ToTensor(),
		download=DOWNLOAD_MNIST
	)

	# print(train_data.train_data.size())
	# print(train_data.train_labels.size())
	# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
	# plt.title('%i' % train_data.train_labels[0])
	# plt.show()

	train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

	test_data = torchvision.datasets.MNIST(
		root='./mnist/',
		train=False,
	)
	# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
	test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000] / 255
	test_y = test_data.test_labels[:2000]

	class CNN(torch.nn.Module):
		def __init__(self):
			super(CNN, self).__init__()
			# input_shape: (1, 28, 28)
			# conv1_shape: (16, 28, 28), pool_shape: (16, 14, 14)
			self.conv1 = torch.nn.Sequential(
				torch.nn.Conv2d(
					in_channels=1,
					out_channels=16,
					kernel_size=5,
					stride=1,
					padding=2,
				),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(kernel_size=2),
			)
			# conv2_shape: (32, 14, 14), pool_shape: (32, 7, 7)
			self.conv2 = torch.nn.Sequential(
				torch.nn.Conv2d(16, 32, 5, 1, 2),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(2)
			)
			# output_shape: (10)
			self.out = torch.nn.Linear(32 * 7 * 7, 10)

		def forward(self, x):
			x = self.conv1(x)
			x = self.conv2(x)
			x = x.view(x.size(0), -1)
			output = self.out(x)
			return output, x

	cnn = CNN()
	print(cnn)

	optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
	loss_func = torch.nn.CrossEntropyLoss()

	for epoch in range(EPOCH):
		for step, (b_x, b_y) in enumerate(train_loader):
			output = cnn(b_x)[0]
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if step % 50 == 0:
				test_output, last_layer = cnn(test_x)
				pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
				accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
				print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

	test_output, _ = cnn(test_x[:10])
	pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
	print(pred_y, 'prediction number')
	print(test_y[:10].numpy(), 'real number')


if __name__ == '__main__':
	start_time = time.time()

	torch_classify()

	end_time = time.time()

	cost_time = end_time - start_time
	print(cost_time)
	m, s = divmod(cost_time, 60)
	h, m = divmod(m, 60)
	print('%02d:%02d:%02d' % (h, m, s))
