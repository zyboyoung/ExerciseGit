import requests

url_root = 'http://www.pythonchallenge.com/pc/def/linkedlist.php?nothing='

def main(code):
	code = requests.get(url_root+code).text.strip().split()
	return code[-1]

if __name__ == '__main__':
	code = '72758'
	while code.isdigit():
		code = main(code)
		print(url_root+str(code))
	print(url_root+str(code))