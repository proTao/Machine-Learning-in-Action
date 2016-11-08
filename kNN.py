
import os
import numpy


def createDataSet():
    group = [[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classifyByKNN(inX, dataSet, labels, k):
    '''
    inX是待分类的样本的特征构成的n维向量
    dataSet是数据集，不含标签，n维
    labels是对应的标签列表
    k是kNN算法参数，代表参考前多少个样本点
    '''
    distance_list = []
    for element in dataSet:
        try:
            distance_list.append(getDistance(inX, element))
        except Exception as e:
            print(inX, element)
    sorted_distance_list = sorted(distance_list)
    # 遍历排序后的距离数组的前k个元素,用dic统计哪个label更多
    dic = {}
    for i in range(k):
        # 当前元素的label
        label = labels[distance_list.index(sorted_distance_list[i])]
        dic[label] = dic.get(label, 0) + 1

    # 返回dic中最大元素
    count = 0
    result = 0
    for i in dic.keys():
        if(dic[i] > count):
            count = dic[i]
            result = i
    return result


def getDistance(vector1, vector2):
    '''
    返回两个n维点的欧几里得距离
    要求两个向量维度相同
    '''
    if(len(vector1) != len(vector2)):
        return None
    else:
        distance = 0
        for i in range(len(vector1)):
            distance += (vector1[i] - vector2[i]) ** 2
        distance = distance ** 0.5
        return distance


def getDataSetFromFile(filename, seperator='\t'):
    '''
    filename是读取文件的名称
    seperator是文件中一行数据的分隔符
    '''
    dataSet = []
    labels = []
    with open(filename, 'r') as f:
        line = f.readline()
        while(line):
            sample = line.split(seperator)
            data = []
            for i in sample[:-1]:
                data.append(float(i))
            dataSet.append(data)
            labels.append(sample[-1][:-1])  # 去掉结尾的换行符
            line = f.readline()

    return (dataSet, labels)


def autoNorm(dataSet):
    '''
    归一化

    返回值的第一项normDataSet是归一化后的矩阵
    返回值的第二项ranges是每一列的值的跨度
    返回值的第三项minVals是每一列的最小值
    '''
    dataSet = numpy.matrix(dataSet)
    # minVals是每一列的最小值，maxVals是每一列的最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = numpy.zeros(dataSet.shape)
    # m是dataSet的行数
    m = dataSet.shape[0]
    normDataSet = dataSet - numpy.tile(minVals, (m, 1))
    normDataSet = normDataSet / numpy.tile(ranges, (m, 1))  # 特征值相除
    return normDataSet.tolist(), ranges.tolist()[0], minVals.tolist()[0]


def datingClassTest():
    '''
    从全部数据集合中选取一部分作为训练集，一部分作为测试集进行测试
    '''
    hoRatio = 0.1
    datingDataMat, datingLabels = getDataSetFromFile(
        "../../机器学习推荐图书/机器学习实战/code/Ch02/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = len(normMat)
    print(m)
    # 测试集的数目
    numTestVecs = int(m * hoRatio)
    print(numTestVecs)
    # 错误率
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classifyByKNN(
            normMat[i], normMat[numTestVecs:m], datingLabels[numTestVecs:m], 1)
        print("the classifier came back with:%s,the real answer is:%s" %
              (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
            # print(classifierResult,datingLabels[i])
    print("the total error rate is:%f" % (errorCount / float(numTestVecs)))


def img2vector(filename):
    returnVect = []
    for i in range(1024):
        returnVect.append(0)
    fr = open(filename, 'r')
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[32 * i + j] = int(lineStr[j])
    fr.close()
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir(
        "../../机器学习推荐图书/机器学习实战/code/Ch02/trainingDigits")
    m = len(trainingFileList)
    trainingMat = []
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat.append(img2vector(
            "../../机器学习推荐图书/机器学习实战/code/Ch02/trainingDigits/%s" % fileNameStr))
    testFileList = os.listdir("../../机器学习推荐图书/机器学习实战/code/Ch02/testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(
            '../../机器学习推荐图书/机器学习实战/code/Ch02/testDigits/%s' % fileNameStr)
        # print(vectorUnderTest)

        classifierResult = classifyByKNN(
            vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with：%d,the real answer is:%d" %
              (classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print('\nthe total number of errors is:%d' % errorCount)
    print("\nthe total error rate is:%f" % (errorCount / float(mTest)))


# datingClassTest()
#(a,b)=getDataSetFromFile("../../机器学习推荐图书/机器学习实战/code/Ch02/datingTestSet2.txt")

handwritingClassTest()
