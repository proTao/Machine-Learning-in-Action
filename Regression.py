'''
Chapter 08
'''

import numpy
import math


def normalize(vec):
    '''
    归一化，具体做法是减去均值再除以方差
    考虑两种可能性：
    1）输入的是普通列表
    2）输入的二维列表，每个元素都是一个样本向量
    '''
    result = []
    if(hasattr(vec[0], "__len__")):
        # 列表元素还是列表时
        mat = numpy.mat(vec).T.tolist()
        # 本来是每个元素是一个样本，转置之后变成每一个元素是一个特征
        for feature in mat:
            result.append(normalize(feature))

        return numpy.mat(result).T.tolist()
    else:
        # 普通列表时，直接算就行
        mean = numpy.mean(vec)
        var = numpy.var(vec)
        if(var == 0 and mean == 1):
            # 特征中人为加的1项，不予处理
            return vec
        result = map(lambda x: (x - mean) / var, vec)
        return list(result)


def loadDataSet(filename):
    '''
    返回的类型是普通列表
    '''
    data_matrix = []
    label = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            lineArr = line[0:-1].split()
            temp_data = [1]
            temp_data.extend(numpy.vectorize(float)(lineArr[0:-1]))
            data_matrix.append(temp_data)
            label.append(float(lineArr[-1]))
    return data_matrix, label


def rssError(yArr, yHatArr):
    '''
    计算预测值与观测值的误差
    '''
    result = []
    for i in range(len(yArr)):
        result.append((yArr[i] - yHatArr[i])**2)
    return sum(result)


def euclideanDistance(x, y):
    '''
    求两个向量的欧几里得距离
    输入值一定要是列表（向量）
    '''
    if len(x) != len(y):
        raise(UserWarning("x,y维度不一致"))
    result = 0
    for i in range(len(x)):
        result += (x[i] - y[i])**2
    return result ** 0.5


def kernel_Guass(x, xi, k=1):
    '''
    高斯核函数
    输入参数必须是向量
    '''
    return math.exp(euclideanDistance(x, xi) ** 2 / -2 * k**2)


def LMSRegression(xArr, yArr):
    '''
    接受参数是普通列表

    LMS梯度下降法(Least-mean-square)
    直接用闭式解的公式
    计算复杂度高的时候可以用梯度下降法

    返回w向量
    '''
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr).T
    xTx = xMat.T * xMat
    if(numpy.linalg.det(xTx)) == 0.0:
        raise(UserWarning("行列式为零，不能求逆"))
    ws = xTx.I * (xMat.T * yMat)
    return ws.T.tolist()[0]


def LWLRegression(x_test, xArr, yArr, kernel_function=kernel_Guass):
    '''
    接受参数是普通列表

    局部加权线性回归
    直接用闭式解的公式
    每次求新的点的回归预测值都需要重新计算
    '''
    # 计算权重
    weight_mat = numpy.eye(len(xArr))
    for i in range(len(xArr)):
        weight_mat[i][i] = kernel_function(x_test, xArr[i])

    # 将输入的列表转化为numpy的矩阵类型
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr).T

    # 检查需要求逆的部分是否有逆
    XTWX = xMat.T * (weight_mat * xMat)
    if(numpy.linalg.det(XTWX) == 0):
        raise(UserWarning("行列式为零，不能求逆"))

    # 直接使用闭式解公式解回归方程
    result_w = XTWX.I * xMat.T * weight_mat * yMat
    return x_test * result_w


def LWLRTest(testArr, xArr, yArr):
    '''
    默认使用高斯核函数
    '''
    m = numpy.shape(testArr)[0]
    print(m)
    yHat = numpy.zeros(m)
    print(yHat)
    for i in range(m):
        yHat[i] = LWLRegression(testArr[i], xArr, yArr, kernel_Guass)
    return yHat


def ridgeRegression(xArr, yArr, k=0.2):
    '''
    岭回归
    k是岭回归的系数

    当k为0时岭矩阵是岭矩阵，得到的结果与普通的回归一致
    随着岭回归系数的从小变大，不重要的系数会缩减为零
    随着k越来越大，越来越多的系数被剔除
    k趋近于无穷大时，所有的系数都为零

    返回w向量
    '''
    # 普通的列表先进行归一化，转化为numpy矩阵类型
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr).T

    # 求括号内需要求逆的部分
    mat = xMat.T * xMat + numpy.eye(len(xArr[1])) * k

    if(numpy.linalg.det(mat) == 0):
        raise(UserWarning("行列式为零，不能求逆"))
    else:
        return (mat.I * (xMat.T * yMat)).T.tolist()[0]


def stageWise(xArr, yArr, eps=0.004, max_iter=20000):
    '''
    前向逐步线性回归算法
    eps是学习速率，max_iter是最大迭代次数
    '''

    # 输入的list列表类型归一化后，转化为numpy矩阵类型
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr).T

    # 初始化参数：当前系数，微调后的临时系数，当前最小误差
    # 必须要初始化为0.0，要是设为0的话，是int型，在numpy中加上0.1后还是0
    w = numpy.mat([0.0] * len(xArr[0])).T
    w_temp = w.copy()
    w_min = w.copy()
    for i in range(max_iter):
        # 在最大迭代次数内进行迭代
        # 最小误差初始化为无穷大
        min_error=1e500
        for j in range(len(w)):
            # 对每一个特征进行启发式的改变
            for change in [1, -1]:
                # 正负方向都进行试探
                w_temp = w.copy()
                w_temp[j] += change * eps
                temp_error = rssError((xMat * w_temp).T.tolist()[0], yArr)
                if(temp_error < min_error):
                    # 如果更改后的参数得到的误差小于现有值，进行更新
                    w_min = w_temp
                    min_error = temp_error
        w = w_min
        print(w.T,min_error)
        # print(w.T,min_error)
    return (w.T, min_error)


#————————————————————测试代码————————————————————
x, y = loadDataSet("../机器学习推荐图书/机器学习实战/code/Ch08/abalone.txt")
# print(x, y)


# 一般线性回归
w = LMSRegression(x, y)
print(w)
yHat = (numpy.mat(w) * numpy.mat(x).T).tolist()[0]
print(rssError(yHat, y))
# 岭回归
w = ridgeRegression(x,y,0.001)
print(w)
yHat = (numpy.mat(w) * numpy.mat(x).T).tolist()[0]
print(rssError(yHat, y))


'''
对岭回归的预测的误差可以发现
可以找到一个岭回归系数来得到最佳的预测（最小的误差）
当然，这个是在训练集上做的误差分析，实际中应该用一个新的集合

for i in range(30):
    yHat = ridgeRegression(x,y,i+1)
    print((yHat.T))
    print(rssError(yHat,y))
'''
# print(numpy.corrcoef(numpy.mat(y), yHat)[0][1])
'''
前向逐步线性回归算法测试
随着学习率的变小，迭代次数的增大，越有更有的系数被挖掘出来
预计当eps极小，迭代次数趋于无穷大时，结果等同于正常的线性回归
反正现在该函数的默认参数是没能跑到最优解。。

但是系数的发掘过程不是单调的
有的系数一开始可能上升的很快到一个比较大的数
然后随着其他的系数的改变，又有可能下降
'''
print(stageWise(x, y,eps=0.1,max_iter=200))
