# coding=UTF-8
'''
Created on 2017年10月16日

@author: pigfish
'''

import math
import random
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

#随机梯度上升算法
def randGradAscent(data, labelVec, num = 150):
    m, n = np.shape(data)
    weights = np.ones((n, 1))
    for j in range(num):
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0, m))
            inZ = np.mat(data[randIndex]) * weights
            classRes = sigmoid(inZ)
            errors = labelVec[randIndex] - classRes
            weights += alpha * np.mat(data[randIndex]).T * float(errors)
    return weights



# if __name__ == '__main__':
#     data, labelVec = loadData()
#     print randGradAscent(data, labelVec)