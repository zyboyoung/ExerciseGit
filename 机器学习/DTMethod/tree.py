# coding=utf-8
# 例子中的数据集格式要求：1.由列表元素组成的列表；2.所有列表元素具有相同长度，且每个列表元素的最后一个元素是标签
from math import log
import operator

# 计算数据集的熵
def calcShannonEnt(dataSet):
	# 得到数据集的实例总数
	numData = len(dataSet)
	# 建立分类的字典
	labelDict = {}
	for eachData in dataSet:
		labelofEachdata = eachData[-1]
		if labelofEachdata not in labelDict.keys():
			labelDict[labelofEachdata] = 0
		labelDict[labelofEachdata] += 1.0
	entropy = 0.0
	for key in labelDict:
		prob = labelDict[key]/numData
		entropy -= prob*log(prob,2)
	return entropy

# 根据不同的特征值对原数据集进行分类
def splitDataSet(dataSet,feature,featValue):
	retDataSet = []
	for eachData in dataSet:
		if eachData[feature] == featValue:
			# 将用于分类的特征从数据集中去除
			tempDataSet = []
			tempDataSet.extend(eachData[:feature])
			tempDataSet.extend(eachData[feature+1:])
			retDataSet.append(tempDataSet)
	return retDataSet

# 找出使得信息增益达到最好的特征
def chooseBestFeature(dataSet):
	# 得到特征数量
	numFeature = len(dataSet[0])-1
	baseEnt = calcShannonEnt(dataSet)
	bestFeature = -1
	bestInformGain = 0.0
	for i in range(numFeature):
		featValueList = [each[i] for each in dataSet]
		featValueSet = set(featValueList)
		newEnt = 0.0
		for j in featValueSet:
			subDataSet = splitDataSet(dataSet,i,j)
			prob = float(len(subDataSet))/len(dataSet)
			newEnt += prob*calcShannonEnt(subDataSet)
		# 计算信息增益，混杂度降低最多的效果最好
		informGain = baseEnt - newEnt
		if informGain > bestInformGain:
			bestInformGain = informGain
			bestFeature = i
	return bestFeature

# 对最后的特征分类结果采取多数的方式决定分类
def majorityCount(classList):
	classDict = {}
	for eachClass in classList:
		if eachClass not in classDict.keys():
			classDict[eachClass] = 0
		classDict[eachClass] += 1
	sortedClassCount = sorted(classDict.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet,labels):
	# labels变量是dataSet中的具体分类标签列表
	classList = [each[-1] for each in dataSet]
	# 所有数据均有相同的类标签
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 遍历完所有特征，dataSet只剩下分类标签
	if len(dataSet[0]) == 1:
		return majorityCount(classList)
	bestFeature = chooseBestFeature(dataSet)
	bestFeatLabel = labels[bestFeature]
	dTree = {bestFeatLabel:{}}
	del(labels[bestFeature])
	featValues = [each[bestFeature] for each in dataSet]
	featValueSet = set(featValues)
	for value in featValueSet:
		subLabels = labels[:]
		dTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
	return dTree

#def createDataSet():
# 	dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
# 	labels = ['no surfacing','flippers']
# 	return dataSet,labels

#data,labels = createDataSet()
#tree = createTree(data,labels)
