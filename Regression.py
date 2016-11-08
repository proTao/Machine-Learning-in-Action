import numpy


def loadDataSet(filename, seperator):
    '''
    filename是读取文件的路径
    seperator是文件中不同特征以及标签的分隔符
    '''
    data_set = []
    label = []
    with open(filename, 'r') as f:
        line = f.readline().strip()
        while(line):
            line_split = line.split(seperator)
            data_set.append(numpy.vectorize(float)(line_split[:-1]))
            label.append(float(line_split[-1]))
            line = f.readline().strip()

    return data_set, label


def standRegres(xArr, yArr):
    # LMS梯度下降法
    # 直接用闭式解的公式
    # 计算复杂度高的时候可以用
    xMat = numpy.mat(xArr)

    yMat = numpy.mat(yArr).T
    xTx = xMat.T * xMat
    if(numpy.linalg.det(xTx)) == 0.0:
        raise(UserWarning)
    ws = xTx.I * (xMat.T * yMat)
    return ws

x, y = loadDataSet("../机器学习推荐图书/机器学习实战/code/Ch08/ex0.txt", "\t")
print(x,y)
print(standRegres(x, y))
