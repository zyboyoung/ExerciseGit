#coding = UTF-8

import randomGrad
import matplotlib.pyplot as plt
import numpy as np

def plotBestFit(weights):
    data, labelVec = randomGrad.loadData()
    n = len(data)
    x1Data = []
    y1Data = []
    x2Data = []
    y2Data = []
    for i in range(n):
        if(int(labelVec[i]) == 1):
            x1Data.append(data[i][1])
            y1Data.append(data[i][2])
        else:
            x2Data.append(data[i][1])
            y2Data.append(data[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1Data, y1Data, s=30, c='red', marker='x', label='class1')
    ax.scatter(x2Data, y2Data, s=30, c='green', marker='o', label='class2')
    plt.xlabel(u'X1')
    plt.ylabel(u'X2')
    plt.legend(loc = 'upper left')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.show()        
              

if __name__ == '__main__':
    data, labelVec = randomGrad.loadData()
    plotBestFit(randomGrad.randGradAscent(data, labelVec))