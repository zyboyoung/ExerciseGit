# coding=UTF-8
from numpy import *

# 词表到向量的转换函数

# 创建实验样本
def loadDataSet():
	wordsList = [
					['my','dog','has','flea','problems','help','please'],
					['maybe','not','take','him','to','dog','park','stupid'],
					['my','dalmation','is','so','cute','I','love','him'],
					['stop','posting','stupid','worthless','garbage'],
					['mr','licks','ate','my','steak','how','to','stop','him'],
					['quit','buying','worthless','dog','food','stupid']
				]
	# 分类标签			
	classVec = [0,1,0,1,0,1]			
	return wordsList,classVec

# 创建包含在所有文档中出现的不重复的列表
def createNewList(dataSet):
	newSet = set([])
	for each in dataSet:
		# 求两个集合的并集
		newSet = newSet | set(each)
	newList = list(newSet)
	return newList

# 将文本转换为词向量(单个)
def words2Vec(newList,inputSet):
	returnVec = [0]*len(newList)
	for each in inputSet:
		if each in newList:
			returnVec[newList.index(each)] += 1
		else:
			print('the word: %s is not in my vocalbulary!' % each)
	return returnVec

# 将所有文本转换为词向量
def trainDoc2Vec(newList,trainDocList):
	trainMatrix = []
	for each in trainDocList:
		trainMatrix.append(words2Vec(newList,each))
	return trainMatrix


# 测试将文本转换为词向量
# if __name__ == '__main__':
# 	wordsList,classVec = loadDataSet()
# 	newList = createNewList(wordsList)
# 	print(newList)
# 	result = trainDoc2Vec(newList,wordsList)
# 	print(result)


# 从词向量计算概率（训练算法）

def trainNB(trainMatrix, trainCategory):
	# 训练样本总数
	numOfTrain = len(trainMatrix)
	# 总特征数目
	numOfWords = len(trainMatrix[0])
	# 二分类中正类的概率，负类的概率为1-p1
	p1 = sum(trainCategory) / float(numOfTrain)
	# 分别创建两个全1向量，用以储存每中类别中每个特征的总数
	vecNum0 = ones(numOfWords)
	vecNum1 = ones(numOfWords)
	# 分别记录两个类别下特征总量，初始化为2
	num0 = 2
	num1 = 2
	for i in range(numOfTrain):
		# 如果该样本属于正类，则将该样本向量与vecNum1相加，最终得到包含正类的所有特征的总数
		if trainCategory[i] == 1:
			vecNum1 += trainMatrix[i]
			num1 += sum(trainMatrix[i])
		# 如果该样本属于负类，则将该样本向量与vecNum0相加，最终得到包含负类的所有特征的总数
		else:
			vecNum0 += trainMatrix[i]
			num0 += sum(trainMatrix[i])
	# 得到两个类别下各特征的概率组成的向量
	vecP1 = log(vecNum1 / float(num1))
	vecP0 = log(vecNum0 / float(num0))

	# 返回的分别是正类、负类中各特征的条件概率，以及正类的概率
	return vecP1,vecP0,p1

		
# 测试计算两分类时正负类下个特征的条件概率以及正类的概率
# if __name__ == '__main__':
# 	wordsList,classVec = loadDataSet()
# 	newList = createNewList(wordsList)
# 	print(newList)
# 	print(classVec)
# 	trainMatrix = trainDoc2Vec(newList,wordsList)
# 	vecP1,vecP0,p1 = trainNB(trainMatrix,classVec)
# 	print(vecP1)
# 	print(vecP0)
# 	print(p1)


# 朴素贝叶斯分类函数
def classify(vect,vecP1,vecP0,p1):
	# 计算P（w | C）× P（C），C=1
	pClass1 = sum(vect*vecP1) + log(p1)
	# 计算P（w | C）× P（C），C=0
	pClass0 = sum(vect*vecP0) + log(1-p1)
	# 对于同一个样本，只需要比较不同类别下的P（w | C）× P（C）的大小，即可分类
	if pClass1 > pClass0:
		return 1
	else:
		return 0

# 分类测试
# def testNB():
# 	wordsList,classVec = loadDataSet()
# 	newList = createNewList(wordsList)
# 	trainMatrix = trainDoc2Vec(newList,wordsList)
# 	vecP1,vecP0,p1 = trainNB(trainMatrix,classVec)
# 	testData1 = ['love','my','dalmation']
# 	testVec1 = words2Vec(newList,testData1)
# 	print testData1,' classified as: ',classify(testVec1,vecP1,vecP0,p1)
# 	testData2 = ['stupid','garbage','dalmation']
# 	testVec2 = words2Vec(newList,testData2)
# 	print testData2,' classified as: ',classify(testVec2,vecP1,vecP0,p1)




		


