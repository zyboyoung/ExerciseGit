#coding=utf-8
import matplotlib.pyplot as plt
# 添加中文字体
from pylab import *
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False

decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle = '<-')

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
	createPlot.axl.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,
		textcoords='axes fraction',va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def createPlot():
	# 创建一个新图形
	fig = plt.figure(1,facecolor='white')
	# 清空绘图区
	fig.clf()
	createPlot.axl = plt.subplot(111,frameon=False)
	plotNode(u'决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
	plotNode(u'叶节点',(0.8,0.1),(0.3,0.8),leafNode)
	plt.show()

# 获取叶节点的数目
def getNumLeafs(tree):
	numLeafs = 0
	# 得到第一个key
	firstStr = tree.keys()[0]
	# 得到该key对应的value
	secondDict = tree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1

# 获取树的层数
def getTreeDepth(tree):
	maxDepth = 0
	firstStr = tree.keys()[0]
	secondDict = tree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth = 1+getNumLeafs(secondDict[key])
		else:
			thisDepth = 1
		if thisDepth>maxDepth:
			maxDepth = thisDepth
	return maxDepth

# 在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
	xMID = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.axl.text(xMid,yMid,txtString)

def plotTree(tree,parentPt,nodeTxt):
	numLeafs = getNumLeafs(tree)
	depth = getTreeDepth(tree)
	firstStr = tree.keys()[0]
	cntrPt = (plotTree.xOff +)






def retrieveTree(i):
	listOfTrees = [{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
					{'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
	return listOfTrees[i]
