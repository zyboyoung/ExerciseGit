def five():
	import os
	import time
	import socket

	# TCP server
	HOST = ''
	PORT = 50007

	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind((HOST, PORT))
		s.listen(1)
		conn, addr = s.accept()
		with conn:
			print('Connected by', addr)
			while True:
				data = conn.recv(1024)
				if not data:
					break
				elif data == b'date':
					conn.sendall(time.ctime().encode('utf-8'))
				elif data == b'os':
					conn.sendall(os.name.encode('utf-8'))
				else:
					conn.sendall(bytes('Input error...', encoding='utf-8'))

	# TCP client
	HOST = 'localhost'
	PORT = 50007

	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
		c.connect((HOST, PORT))
		while True:
			data = input('> ').encode('utf-8')
			c.sendall(data)
			data = c.recv(1024)
			if not data:
				break
			print('Received:', str(data, encoding = 'utf-8'))

def seven():
	import socket
	import os

	# # TCP server
	# HOST = ''
	# PORT = 9999
	# BUFSIZ = 1024
	#
	# tcpSerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# tcpSerSock.bind((HOST, PORT))
	# tcpSerSock.listen(1)
	#
	# while True:
	# 	print('...waiting for connection...')
	# 	tcpCliSock, addr = tcpSerSock.accept()
	# 	print('...connected from:', addr)
	#
	# 	while True:
	# 		data = tcpCliSock.recv(BUFSIZ)
	# 		if data == b'q' or data == b'quit':
	# 			tcpCliSock.close()
	# 		else:
	# 			print('%s said: %s' % (addr, data.decode('utf-8')))
	# 		message = ''
	# 		while not message:
	# 			message = input('> ').encode('utf-8')
	# 		tcpCliSock.send(message)
	# 		data = None
	# tcpCliSock.close()

	# TCP client
	HOST = 'localhost'
	PORT = 9999

	cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	cs.connect((HOST, PORT))

	while True:
		data = input('> ').encode('utf-8')
		if not data:
			continue
		cs.sendall(data)
		re_data = cs.recv(1024)
		print('Received:', re_data.decode('utf-8'))
	cs.close()


if __name__ == '__main__':
	seven()