import re

def One(str):
	pattern = 'bat|bit|but|hat|hit|hut'
	print(re.search(pattern, str).group())

def Two(str):
	print(re.split(' ', str))

def Three(str):
	pattern = ', '
	print(re.split(pattern, str))

def Four(str):
	pattern = '[A-Za-z_]+[\w_]+'
	print(re.match(pattern, str).group())

def Five(str):
	pattern = '(\w+[ ]?)*\w+'
	print(re.match(pattern, str).group())

def Six(str):
	pattern = '^www.*\.(com|edu|net)'
	print(re.search(pattern, str).group())

def Seven(str):
	pattern = '\d+'
	print(re.findall(pattern, str))

def Eight(str):
	pattern = '\d+'
	print(re.findall(pattern, str))

def Nine(str):
	pattern = '\d+(.\d*)?'
	print(re.match(pattern, str).grouop())

def Thirty():


if __name__ == '__main__':
	Three('Mr.sd, Mr.')