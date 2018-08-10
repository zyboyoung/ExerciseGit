# coding=utf-8

import bayes
import re
import random

def textParse(string):
	listToken = re.split(r'\W',string)
	return [each.lower() for each in listToken if len(each)>2]

def emailTest():
	# 储存所有文档，以一个列表的形式保存
	allDoclist = []
	# 储存所有的邮件，每个邮件属于一个列表
	allEmail = []
	# 储存每封邮件的类别
	classList = []
	for i in range(1,26):
		with open('../email/spam/%d.txt' % i) as text:
			wordList = textParse(text.read())
			allDoclist.append(wordList)
			allEmail.extend(wordList)
			classList.append(1)
		with open('../email/ham/%d.txt' % i) as text:
			wordList = textParse(text.read())
			allDoclist.append(wordList)
			allEmail.extend(wordList)
			classList.append(0)
	# 将所有文档中的单词去除重复
	newWordsList = bayes.createNewList(allDoclist)
	# 邮件总数为50，从中划分测试集和训练集
	trainingSet = range(50)
	testSet = []
	for i in range(10):
		# 随机数最开始的浮点数没有取到首尾，取Int后有首尾0，最后
		randIndex = int(random.uniform(0,len(trainingSet)))
		# 添加测试集
		testSet.append(trainingSet[randIndex])
		# 删除训练集中对应部分
		del trainingSet[randIndex]
	trainingVec = []
	trainClass = []
	for trainingIndex in trainingSet:
		trainingVec.append(bayes.words2Vec(newWordsList,allDoclist[trainingIndex]))
		trainClass.append(classList[trainingIndex])
	vecP1,vecP0,p1 = bayes.trainNB(trainingVec, trainClass)
	errorCount = 0
	for testIndex in testSet:
		testVec =  bayes.words2Vec(newWordsList,allDoclist[testIndex])
		if bayes.classify(testVec, vecP1, vecP0, p1) != classList[testIndex]:
			errorCount += 1
			print 'classification error: ',allDoclist[testIndex]
	print 'The error rate is: ',float(errorCount)/len(testSet)
if __name__ == '__main__':
	emailTest()