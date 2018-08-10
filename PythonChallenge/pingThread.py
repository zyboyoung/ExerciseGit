import threading
import queue
import os


class Pinger(threading.Thread):
	def __init__(self, queue, pingIp, pingCount = 1):
		threading.Thread.__init__(self)
		self.queue = queue
		self.pingIp = pingIp
		self.pingCount = 1
	def run(self):
		pingResult = os.popen('ping -n' + ' ' + str(self.pingCount) + ' ' +self.pingIp).read()
		if '无法访问目标主机' not in pingResult:
			print(self.pingIp,'\t is online')
		self.queue.get()

class createpinger:
	def __init__(self, queue, pingIpParagraph, allcount = 255, pingCount = 1):
		self.queue = queue
		self.pingIpParagraph = pingIpParagraph
		self.allcount = allcount
		self.pingCount = 1
		self.create()

	def create(self):
		for i in range(1, self.allcount+1):
			self.queue.put(i)
			Pinger(self.queue, self.pingIpParagraph+str(i), self.pingCount).start()

if __name__ == '__main__':
	for i in range(0, 255):
		createpinger(queue.Queue(100), '111.222.'+str(i)+'.')