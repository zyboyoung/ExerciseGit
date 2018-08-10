import re
import urllib.request

def getData():
	text = urllib.request.urlopen('http://www.pythonchallenge.com/pc/def/equality.html')
	data = text.read()
	return data

if __name__ == '__main__':
	result = re.findall('[^A-Z][A-Z]{3}([a-z])[A-Z]{3}[^A-Z]', str(getData()))
	print(''.join(result))