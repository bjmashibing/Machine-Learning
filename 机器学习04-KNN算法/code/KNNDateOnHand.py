# coding:utf-8
'''
Created on 2020年1月11日
@author: zfg
'''

import numpy as np
import operator
import matplotlib.pyplot as plt
from array import array
from matplotlib.font_manager import FontProperties


def classify(normData, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(normData, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances ** 0.5
    sortedDistIndicies = distance.argsort()
    #     classCount保存的K是魅力类型   V:在K个近邻中某一个类型的次数
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    #     readlines:是一次性将这个文本的内容全部加载到内存中(列表)
    arrayOflines = fr.readlines()
    numOfLines = len(arrayOflines)
    returnMat = np.zeros((numOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOflines:
        line = line.strip()
        print(line.split('\t'))
        listFromline = list(map(float, line.split('\t')))

        returnMat[index, :] = listFromline[0:3]
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector


'''
    将训练集中的数据进行归一化
    归一化的目的：
        训练集中飞行公里数这一维度中的值是非常大，那么这个纬度值对于最终的计算结果(两点的距离)影响是非常大，
        远远超过其他的两个维度对于最终结果的影响
    实际约会姑娘认为这三个特征是同等重要的
    下面使用最大最小值归一化的方式将训练集中的数据进行归一化
'''


def autoNorm(dataSet):
    #     dataSet.min(0)   代表的是统计这个矩阵中每一列的最小值     返回值是一个矩阵1*3矩阵
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    #     normDataSet存储归一化后的数据
    normDataSet = np.zeros(np.shape(dataSet))
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('../../../data/datingTestSet2.txt')

    # 归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)

    #     shape获取矩阵的行数以及列数，以二元组的形式返回的
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], \
                                    datingLabels[numTestVecs:m], 4)
        print('模型预测值: %d ,真实值 : %d' \
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    errorRate = errorCount / float(numTestVecs)
    print('正确率 : %f' % (1 - errorRate))
    return 1 - errorRate


'''
    拿到每条样本的飞行里程数和玩视频游戏所消耗的事件百分比这两个维度的值，使用散点图
'''


def createScatterDiagram():
    datingDataMat, datingLabels = file2matrix('../../../data/datingTestSet2.txt')
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    fig = plt.figure()
    axes = plt.subplot(111)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    for i in range(len(datingLabels)):
        if datingLabels[i] == 1:  # 不喜欢
            type1_x.append(datingDataMat[i][0])
            type1_y.append(datingDataMat[i][1])

        if datingLabels[i] == 2:  # 魅力一般
            type2_x.append(datingDataMat[i][0])
            type2_y.append(datingDataMat[i][1])

        if datingLabels[i] == 3:  # 极具魅力
            type3_x.append(datingDataMat[i][0])
            type3_y.append(datingDataMat[i][1])

    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')
    plt.xlabel(u'每年飞行里程数')
    plt.ylabel(u'玩视频游戏所消耗的事件百分比')
    # axes.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2)
    plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1], c=datingLabels)
    plt.show()


def classifyperson():
    resultList = ['没感觉', '看起来还行', '极具魅力']
    input_man = [50000, 8, 9.5]
    datingDataMat, datingLabels = file2matrix('../../../data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    result = classify((input_man - minVals) / ranges, normMat, datingLabels, 10)
    print('你即将约会的人是:', resultList[result - 1])


if __name__ == '__main__':
    #     createScatterDiagram观察数据的分布情况
    # createScatterDiagram()
    acc = datingClassTest()
    if (acc > 0.9):
        classifyperson()
