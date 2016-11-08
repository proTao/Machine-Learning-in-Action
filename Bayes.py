import random
import math
import numpy


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(data_set):
    '''
    dataset是一个列表的列表
    返回所有出现过的单词的集合
    以列表形式返回
    '''
    vocab_list = set()
    for document in data_set:
        vocab_list = vocab_list.union(document)
    return list(vocab_list)

def wordSet2Vector(vocab_list, input_set):
    '''
    返回一个文档集合在单词表中的对应的二值向量
    如果输入的input_set中含有词汇表中不存在的单词则抛出异常
    '''
    returnVec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            returnVec[vocab_list.index(word)] = 1
        else:
            raise(UserWarning)
    return returnVec

def BayesTrainer(train_matrix, train_labels):
    feature_num = len(train_matrix[0])
    sample_num = len(train_labels)

    # 分别统计两个类中各个单词出现的频数，为了防止零概率的出现，初值赋为零
    wordcount_A = numpy.ones(feature_num)
    wordcount_B = numpy.ones(feature_num)

    #两个类中的全部单词数
    count_A=0
    count_B=0

    # 先验概率
    # 因为是二类分类器，train_labels中的标签是0,1，对标签向量的求和的值是1类的个数
    prob_priorA = sum(train_labels) / sample_num
    prob_priorB = 1 - prob_priorA

    # 遍历数据集，统计单词数
    for i in range(len(train_labels)):
        print(wordcount_A)
        print(train_matrix[i])
        if(train_labels[i] == 1):
            wordcount_A += train_matrix[i]
            count_A += sum(train_matrix[i])
        else:
            wordcount_B += train_matrix[i]
            count_B += sum(train_matrix[i])

    # 转化单词频数为单词频率，拟合概率
    # 太多小数的相乘可能会结果下溢，用log进行平滑
    # numpy.vectorize函数的作用是把接受一个参数的函数转化为接受一个list的函数，对list中每个元素依次处理
    prob_wordA = numpy.vectorize(math.log2)(wordcount_A / count_A)
    prob_wordB = numpy.vectorize(math.log2)(wordcount_B / count_B)

    # 测试用输出
    print(prob_wordA)
    print(prob_wordB)
    print(prob_priorA)
    print(prob_priorB)

    return (prob_wordA, prob_wordB, prob_priorA, prob_priorB)

def BayesClassifier(test_vector, prob_wordA, prob_wordB, prob_priorA, prob_priorB):
    pA=sum(test_vector * prob_wordA) + math.log2(prob_priorA)
    pB=sum(test_vector * prob_wordB) + math.log2(prob_priorB)
    if(pA > pB):
        return 1
    else:
        return 0

def textParse(text):
    '''
    输入一段英文文本
    分词，去掉长度小于3的单词
    返回词列表
    '''
    import re
    token_list = re.split(r"\W*", text)
    return [token.lower() for token in token_list if len(token)>2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        wordList = textParse(open("../../机器学习推荐图书/机器学习实战/code/Ch04/email/spam/%d.txt"%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open("../../机器学习推荐图书/机器学习实战/code/Ch04/email/ham/%d.txt"%1).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocab_list=createVocabList(docList)
    trainingSet=[i for i in range(50)]
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(wordSet2Vector(vocab_list,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    pOV,plV,pSpam,pHam=BayesTrainer(trainMat,trainClasses)
    errorCount=0
    for docIndex in testSet:
        wordVector=wordSet2Vector(vocab_list,docList[docIndex])
        if BayesClassifier(wordVector,pOV,plV,pSpam,pHam) != classList[docIndex]:
            errorCount+= 1
    print('the error rate is: ',float(errorCount)/len(testSet))







''' 第一部分测试代码
data_set, label = loadDataSet()
vocab_list = createVocabList(data_set)
data_matrix=[]
for data in data_set:
    data_matrix.append(wordSet2Vector(vocab_list,data))
prob_wordA, prob_wordB, prob_priorA, prob_priorB = BayesTrainer(data_matrix, label)
print(BayesClassifier(wordSet2Vector(vocab_list,["love","my","dalmation"]),prob_wordA, prob_wordB, prob_priorA, prob_priorB))
print(BayesClassifier(wordSet2Vector(vocab_list,["stupid","garbage"]),prob_wordA, prob_wordB, prob_priorA, prob_priorB))
'''

spamTest()



