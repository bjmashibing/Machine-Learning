# coding:utf-8
'''
Created on 2020年1月11日

@author: root
'''
from sklearn.neighbors.unsupervised import NearestNeighbors
import numpy as np
from com.msb.knn.KNNDateOnHand import *

datingDataMat, datingLabels = file2matrix('../../../data/datingTestSet2.txt')
normMat, ranges, minVals = autoNorm(datingDataMat)

nbrs = NearestNeighbors(n_neighbors=10).fit(normMat)
input_man = [[50000, 8, 9.5]]
S = (input_man - minVals) / ranges
distances, indices = nbrs.kneighbors(S)
# classCount   K：类别名    V：这个类别中的样本出现的次数

classCount = {}
for i in range(10):
    voteLabel = datingLabels[indices[0][i]]
    classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
resultList = ['没感觉', '看起来还行', '极具魅力']
print(resultList[sortedClassCount[0][0] - 1])
