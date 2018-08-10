import pickle
import requests
import urllib.request

url = 'http://www.pythonchallenge.com/pc/def/banner.p'
data = urllib.request.urlopen(url).readlines()
# data = requests.get(url).content
print(data)
# data2 = str(requests.get(url).content).split(r'\n')
# # data3 = []
# # for i in data2:
# # 	data3.append(i)
# print(data2)
data = pickle.loads(b''.join(data))
print(data)

if __name__ == '__main__':
	for r in data:
		result = ''
		for s in r:
			result += s[0]*s[1]
		print(result)
