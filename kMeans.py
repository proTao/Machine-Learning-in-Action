from myTools import *
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


def getRange(dataset):
    '''
    获取dataset的数据在各个维度（特征）上的跨度

    输入一个二维列表，存放数据
    输入一个二维列表，每个元素是该维度上的最小值和最大值
    '''
    datamat = np.mat(dataset).T.tolist()
    dimen_range = []
    for dimension in datamat:
        max_value = max(dimension)
        min_value = min(dimension)
        dimen_range.append([min_value, max_value])
    return dimen_range


def randCenter(dimension_range, k):
    '''
    在dimension_range的范围内生成k个随机点

    输入dimension_range是一个二维list
    返回值是二维list
    '''
    # 获取数据集维度
    import random
    dimen_num = len(dimension_range)
    result = []
    for i in range(k):
        # 生成k个点
        randpoint = []
        for j in dimension_range:
            randpoint.append(random.uniform(j[0], j[1]))
        result.append(randpoint)
    return result


def getMinDistance(point, centers, distance_func):
    '''
    返回point到其他几个点的最近的点与最小的距离

    返回值：（最近点在centers中的索引，最小距离）
    '''
    # 初始化参数为最近的点是第0个点
    i = 0
    close_point = i
    min_distance = distance_func(point, centers[i])

    i += 1  # 跳过第0个点不用检查
    while(i < len(centers)):
        temp_distance = distance_func(point, centers[i])
        if(temp_distance < min_distance):
            close_point = i
            min_distance = temp_distance
        i += 1
    return [close_point, min_distance]


def getMeanVector(dataset):
    '''
    返回数据集的均值向量
    '''
    datamat = np.mat(dataset)
    feature_num = len(dataset.tolist()[0])
    
    mean_vector = []
    for i in range(feature_num):
        mean_vector.append(np.mean(datamat[:, i]))
    return mean_vector


def KMeans(dataset, k, centers, distance_func=EuclideanDistance):
    '''
    K均值聚类算法,由于用到了绘图，所以目前只支持二维数据

    data是二维列表
    k是最后聚簇的数目
    centers是二维向量，提供初始中心店
    dictance_func是可选参数，求距离的算法

    返回类型：二维列表
    '''
    datamat = np.mat(dataset)

    if(k != len(centers)):
        raise(UserWarning("中心点的数目不正确"))
    feature_num = len(centers[0])

    # 标志位，聚簇点不再改变时停止循环
    is_change = True

    # 初始化距离矩阵，存放各个点最近的中心点以及最近距离
    distance = []
    for point in dataset:
        distance.append(getMinDistance(point, centers, distance_func))
    distance = np.mat(distance)
    
    color_spot=["b.","g.","r.","y."]
    color_circle=["bo","go","ro","yo"]
    for i in range(k):
        sub_datamat=datamat[np.nonzero(distance[:, 0] == i)[0]]
        x1=np.mat(sub_datamat)[:,0].T.tolist()[0]
        y1=np.mat(sub_datamat)[:,1].T.tolist()[0]
        plt.plot(x1,y1,color_spot[i])
        plt.plot(centers[i][0],centers[i][1],color_circle[i])
    plt.show()

    while(is_change):
        is_change = False
        # plt.cla()
        # plt.scatter(datamat[:,0],datamat[:,1],marker=".")
        # plt.scatter(np.mat(center)[:,0],np.mat(center)[:,1],marker="+")

        # 更新中心点
        for i in range(k):
            try:
                centers[i] = getMeanVector(datamat[np.nonzero(distance[:, 0] == i)[0]])
            except:
                # 没有数据点离这个点近
                pass

        # 更新距离矩阵
        for i in range(len(dataset)):
            # print(dataset[i])
            # print(centers)
            point_distance = getMinDistance(dataset[i], centers, distance_func)
            print()
            print(i)
            print(dataset[i])
            print(point_distance)
            print(distance[i])
            if point_distance[1] < distance[i, 1] or point_distance[0] != distance[i, 0]:
                # 所属类型更改或者最小距离变小都需要更新
                distance[i] = point_distance
                print("--------->"+str(distance[i]))
                is_change = True

        
        for i in range(k):
            sub_datamat=datamat[np.nonzero(distance[:, 0] == i)[0]]
            x1=np.mat(sub_datamat)[:,0].T.tolist()[0]
            y1=np.mat(sub_datamat)[:,1].T.tolist()[0]
            plt.plot(x1,y1,color_spot[i])
            plt.plot(centers[i][0],centers[i][1],color_circle[i])
        
        print(centers)
        plt.show()
    
    for i in range(len(distance)):
        print(i)
        print(distance[i])
    # print()
    # print("datamat1")
    # print(datamat1)
    # print()
    # print("datamat2")
    # print(datamat2)
    # print()
    # print("datamat3")
    # print(datamat3)

    return centers


dataset = loadDataSet(
    "../机器学习推荐图书/机器学习实战/code/Ch10/testSet1.txt", with_label=False)
k = 4
init_centers = randCenter(getRange(dataset), k)
print(init_centers)


centers = KMeans(dataset, k, init_centers)
# point=[2,-1]
# print(getMinDistance(point,[[-1,1],[1,1],[-1,-1],[1,-1]],EuclideanDistance))
