# coding=UTF-8
'''
Created on 2017年10月16日

@author: pigfish
'''

import math
import numpy as np

#读取数据，构建数据和标签矩阵
def loadData():
    data = []
    labelVec = []
    with open('../testSet.txt') as dataTxt:
        for line in dataTxt:
            line = line.strip().split()
            data.append([1.0, float(line[0]), float(line[1])])
            labelVec.append(int(line[2]))
    return data, labelVec

#计算Sigmoid函数值
def sigmoid(inX):
    if type(inX) == np.matrixlib.defmatrix.matrix:
        return np.mat([1.0/(1 + math.exp(-i)) for i in inX]).T
    else:
        return 1.0/(1 + math.exp(-inX))
    
#梯度上升算法
def gradAscent(data, labelVec):
    dataMat = np.mat(data)
    labelMat = np.mat(labelVec).T
    m, n = np.shape(dataMat)
    alpha = 0.001
    maxLoops = 500
    weights = np.ones((n, 1))
    for i in range(maxLoops):
        inMat = dataMat * weights
        classMat = sigmoid(inMat)
        errors = labelMat - classMat
        weights += alpha * dataMat.T * errors
    return weights

# if __name__ == '__main__':
#     data, labelVec = loadData()
#     print gradAscent(data, labelVec)