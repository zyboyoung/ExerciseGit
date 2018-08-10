#coding=UTF-8
from numpy import *
from os import listdir
import operator
import matplotlib
import matplotlib as plt

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	# inX表示用于分类的输入向量，dataSet表示输入的训练样本集，labels表示标签向量
	# k表示选择的最近邻数目，labels的元素数目和dataSet的行数相同
	dataSetSize = dataSet.shape[0]
	# 得到dataSet的第一维度的长度，也就是行数
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	# 将输入向量扩展到dataSet相同的维数，并作矩阵的减法
	sqDiffMat = diffMat**2
	# 计算欧几里得距离
	sqDistances = sqDiffMat.sum(axis=1)
	# 将矩阵的每一行相加，也就是得到平方和
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()

	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


# 改进约会网站
def file2matrix(filename):
	fr = open(filename)
	# 得到文件所有行并返回列表，每一行作为列表中的一个元素
	arrayOfLines = fr.readlines()
	# 得到文件的行数
	numberOfLines = len(arrayOfLines)
	# 参数为一个列表，其中分别是矩阵的行数和列数，矩阵是零矩阵
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	# 选取列表中的每个元素，变量名为line
	for line in arrayOfLines:
		# 截取掉所有的回车字符
		line = line.strip()
		# 将截取回车后得到的整行数据分割成一个元素列表，原来的一行成为了一个新的列表，以\t作为分隔符，将列表中的元素分开
		listFromLine = line.split('\t')
		# 将列表中的前三个元素分别复制到矩阵中
		returnMat[index,:]=listFromLine[0:3]
		# 将列表中的最后一列元素也就是标签存储到向量中
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

# 特征值的归一化
def autoNorm(dataSet):
	# 从列中选取最小值
	minValue = dataSet.min(0)
	# 从列中选取最大值
	maxValue = dataSet.max(0)
	ranges = maxValue - minValue
	# 生成给定形状，也就是与dataSet相同的矩阵，不过是零矩阵
	normDataSet = zeros(shape(dataSet))
	# 读取第一维度的长度，行数
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minValue,(m,1))
	# 在Numpy库中可以直接相除，得到的是每个数值对应相除的结果
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet, ranges, minValue

# 对约会分类器的评估
def datingClassTest():
	testRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*testRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print('the classifierResult came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
		if classifierResult!=datingLabels[i]:
			errorCount += 1.0
	print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

# 约会网站预测函数
def classifyPerson():
	resultList = ['not at all','in small doses','in large doses']
	# raw_inout()允许用户输入文本行命令并返回用户输入的命令
	percentTats = float(raw_input('percentage of time spent playing video games?'))
	ffMiles = float(raw_input('frequent flier miles earned per year?'))
	iceCream = float(raw_input('liters of ice cream consumed per year?'))
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([percentTats,ffMiles,iceCream])
	classifierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)
	print('You will probably like this person: '+resultList[classifierResult-1])

# 将图像转换为测试向量
def img2vector(filename):
	returnVect = zeros((1,1024))
	with open(filename) as fr:
		for i in range(32):
			lineStr = fr.readline()
			for j in range(32):
				returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	# listdir()可以给出指定目录的文件名
	traingingFileList = listdir('trainingDigits')
	# 获得文件总数
	m = len(traingingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = traingingFileList[i]
		# 将后缀名txt去掉
		fileStr = fileNameStr.split('.')[0]
		# 提取出样本的标签
		classNumStr = int(fileStr.split('_')[0])
		# 得到所有样本集的标签
		hwLabels.append(classNumStr)
		# 得到样本数据矩阵
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	n = len(testFileList)
	errorCount = 0.0
	for i in range(n):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print('the classifier came back with: %d, the real answer is: %d' % (classifierResult,classNumStr))
		if classNumStr!=classifierResult:
			errorCount += 1.0
	print('the total number of errors is: %d' % errorCount)
	print('the total error rate is: %f' % (errorCount / n))

if __name__ == '__main__':
	handwritingClassTest()