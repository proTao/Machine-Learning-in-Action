import math
import numpy
import random


def loadDataSet(filename):
    data_matrix = []
    label = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            lineArr = line[0:-1].split()
            temp_data=[1]
            temp_data.extend(numpy.vectorize(float)(lineArr[0:-1]))
            data_matrix.append(temp_data)
            label.append(float(lineArr[-1]))
    return data_matrix, label

def loadDataSet_frombook(filename):
    dataMat=[]
    labelMat=[]
    fr=open(filename,'r')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(x):
    '''
    用sigmoid函数模拟越阶函数
    '''
    return 1 / (1 + math.exp(-x))

def sign(x):
    '''
    阶跃函数
    '''
    if(x>0):
        return 1.0
    else:
        return 0
def batchGradDescent(data_matrix, label, function):
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
        h = numpy.vectorize(function)(data_matrix * weights)
        error = h - label
        weights = weights - step * data_matrix.transpose() * error
    return weights


def stocGradDescent(data_matrix, label, function):
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
            randIndex = int(random.uniform(0, m))  # 不是周期性的选取，而是随机性的
            step = 4 / (1 + i + j) + 0.01
            h = function(data_matrix[i] * weights)
            error = h - label[i]
            weights = weights - step * data_matrix[i].transpose() * error
    return weights

# 分类样本 Ch05/testSet.txt
data_set, label = loadDataSet("../机器学习推荐图书/机器学习实战/code/Ch05/testSet.txt")
print(batchGradDescent(data_set, label, sigmoid))

# 分类样本 Ch05/testSet.txt
# 这个在cs229课程中，用跃阶函数代替sigmoid函数被称为perception
data_set, label = loadDataSet("../机器学习推荐图书/机器学习实战/code/Ch05/testSet.txt")
print(batchGradDescent(data_set, label, sign))

# 回归样本 Ch08/ex0.txt
data_set, label = loadDataSet("../机器学习推荐图书/机器学习实战/code/Ch08/ex0.txt")
print(batchGradDescent(data_set, label, lambda x:x))
