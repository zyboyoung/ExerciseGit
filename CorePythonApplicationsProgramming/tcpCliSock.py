from socket import *

HOST = 'localhost'
PORT = 21560
BUFSIZ = 1024
ADDR = (HOST, PORT)

tcpCliSock = socket()
tcpCliSock.connect(ADDR)

while(1):
	data = bytes(input('> '), encoding='utf-8')
	if not data:
		break
	tcpCliSock.send(data)
	data = tcpCliSock.recv(BUFSIZ)
	if not data:
		break
	print(data.decode('utf-8'))

tcpCliSock.close()
