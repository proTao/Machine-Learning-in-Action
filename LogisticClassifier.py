import math
import numpy
import random

def loadDataSet():
    data_matrix = []
    label = []
    with open("../机器学习推荐图书/机器学习实战/code/Ch05/testSet.txt", 'r') as f:
        for line in f.readlines():
            lineArr = line[0:-1].split()
            data_matrix.append([1, float(lineArr[0]), float(lineArr[1])])
            label.append(int(lineArr[-1]))
    return data_matrix, label


def sigmoid(x):
    '''
    用sigmoid函数模拟越阶函数
    '''
    return 1 / (1 + math.exp(-x))


def batchGradDescent(data_matrix, label):
    '''
    梯度下降法
    '''
    data_matrix = numpy.mat(data_matrix)
    label = numpy.mat(label).transpose()
    m, n = numpy.shape(data_matrix)
    step = 0.001  # 学习速率
    max_cycles = 500  # 最大循环次数
    weights = numpy.ones((n, 1))
    for k in range(max_cycles):
        # Ng讲的直接的梯度法没有这个sigmoid函数，在logistic回归中加入这个logistic函数
        h = numpy.vectorize(sigmoid)(data_matrix * weights)
        error = h - label
        weights = weights - step * data_matrix.transpose() * error
    return weights

def stocGradDescent(data_matrix, label):
    '''
    梯度下降法
    '''
    data_matrix = numpy.mat(data_matrix)
    label = numpy.mat(label).transpose()
    m, n = numpy.shape(data_matrix)
    max_cycles = 20  # 最大循环次数
    weights = numpy.ones((n, 1))
    for j in range(max_cycles):
        # Ng讲的直接的梯度法没有这个sigmoid函数，在logistic回归中加入这个logistic函数
        for i in range(m):
            randIndex=int(random.uniform(0,m)) # 不是周期性的选取，而是随机性的
            step=4/(1+i+j)+0.01
            h = sigmoid(data_matrix[i] * weights)
            error = h - label[i]
            weights = weights - step * data_matrix[i].transpose() * error
    return weights


data_set, label = loadDataSet()
print(stocGradDescent(data_set, label))
