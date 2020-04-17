#coding:utf-8  

'''
Created on 2020年1月20日
@author: zfg
'''

import numpy as np
import math as mt
from collections import defaultdict

class DecisionTree(object):
    def __init__(self):
        pass

#   计算概率
    def _getDistribution(self,dataArray):
        # dict  map
        distribution = defaultdict(float)
        m, n = np.shape(dataArray)
        for line in dataArray:
            print(line[-1])
            distribution[line[-1]] += 1.0/m
        # 每一个分类号出现的概率  yes 5/14  no 9/14
        return distribution

#   计算信息熵
    def _entropy(self, dataArray):
        ent = 0.0
        distribution=self._getDistribution(dataArray)

        for key, prob in distribution.items():
            ent -= prob * mt.log(prob, 2)
        return ent

    def _conditionEntropy(self, dataArray, colIdx):
        valueCnt = defaultdict(int)
        m, n = np.shape(dataArray)
        # 条件熵
        condEnt = 0.0
        uniqueValues = np.unique(dataArray[:, colIdx])
        for oneValue in uniqueValues:
            oneData = dataArray[dataArray[:, colIdx] == oneValue]
            # 信息熵
            oneEnt = self._entropy(oneData)
            # 第一列值为teenager的概率
            prob = float(np.shape(oneData)[0]) / m
            # 概率*信息熵
            condEnt += prob * oneEnt
        return condEnt

    def _infoGain(self, dataArray, colIdx, baseEnt):
        condEnt = self._conditionEntropy(dataArray, colIdx)
        # 信息增益
        return baseEnt-condEnt

    def _chooseBestProp(self,dataArray):
        m, n = np.shape(dataArray)
        bestProp = -1
        bestInfoGain = 0
        # 计算分类号信息熵
        baseEnt = self._entropy(dataArray)
        # [0-4)
        for i in range(n-1):
            # 计算已知第一列数据的信息熵  条件熵
            infoGain=self._infoGain(dataArray, i, baseEnt)
            if infoGain > bestInfoGain:
                bestProp=i
                bestInfoGain=infoGain
        return bestProp

    def _splitData(self,dataArray,colIdx,splitValue):
        m, n = np.shape(dataArray)

        cols = np.array(range(n)) != colIdx
        rows = (dataArray[:, colIdx] == splitValue)
        print(rows)

        # data=dataArray[rows,:][:,cols]
        #ix_  取rows中指定的行，取cols中指定的列   花式索引
        data = dataArray[np.ix_(rows, cols)]
        return data

    def createTree(self, dataArray):
        # 获取集合形状 (m,n)
        m, n = np.shape(dataArray)
        if len(np.unique(dataArray[:, -1])) == 1:
            return (dataArray[0, -1], 1.0)
        if n == 2:
            distribution = self._getDistribution(dataArray)
            sortProb = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            return sortProb
        rootNode = {}
        # 选择分类条件的     信息增益  0
        bestPropIdx = self._chooseBestProp(dataArray)

        # 树
        rootNode[bestPropIdx] = {}
        uniqValues = np.unique(dataArray[:, bestPropIdx])
        # 根据第一列的数据来分类
        for oneValue in uniqValues:
            splitDataArray = self._splitData(dataArray, bestPropIdx, oneValue)
            # 要不要把分类出来的这堆数据  进行再次切割   信息熵判断一下
            rootNode[bestPropIdx][oneValue] = self.createTree(splitDataArray)
        return rootNode
    
def loadData():
    # 矩阵
    dataMat = []                 
    fr = open("decisiontree.txt")
#     readlines他会一次性将decisiontree.txt文件全部加载到内存的列表中
    lines = fr.readlines()
    for line in lines:
        curLine = line.strip().split('\t')
        dataMat.append(curLine)
    return dataMat


if __name__ == '__main__':
    data = loadData()
    dataarray = np.array(data)
    dt = DecisionTree()
    tree = dt.createTree(dataarray)
    print(tree)
