import numpy

def binSplitDataSet(dataSet, feature, value):
    '''
    切分矩阵
    dataset是二维列表
    
    把dataset中列索引是feature的大于value的行放到mat0中
    其余的放到mat1中
    具体怎么工作的，这代码有点简洁，用print看吧
    '''
    '''
    print(type(dataSet))
    print(dataSet[:, feature])
    print(dataSet[:, feature] < value)
    print(type(numpy.nonzero(dataSet[:, feature] < value)[0]))
    print(dataSet[numpy.nonzero(dataSet[:, feature] < value)[0], :])
    '''
    mat=numpy.mat(dataSet)
    mat0 = mat[numpy.nonzero(mat[:, feature] > value)[0], :][0]
    mat1 = mat[numpy.nonzero(mat[:, feature] <= value)[0], :][0]
    return mat0, mat1
a = binSplitDataSet(numpy.mat(numpy.eye(4)), 1, 0.5)
print(a[0])
print()
print(a[1])