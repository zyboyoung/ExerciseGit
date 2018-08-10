from socket import *

HOST = '127.0.0.1'
PORT = 21241
BUFSIZ = 1024
ADDR = (HOST, PORT)

while True:
	tcpCliSock = socket()
	tcpCliSock.connect(ADDR)
	data = input('> ')
	if not data:
		break
	tcpCliSock.send(('%s\r\n' % data).encode('utf-8'))
	data = tcpCliSock.recv(BUFSIZ)
	if not data:
		break
	print(data.strip().decode('utf-8'))
	tcpCliSock.close()