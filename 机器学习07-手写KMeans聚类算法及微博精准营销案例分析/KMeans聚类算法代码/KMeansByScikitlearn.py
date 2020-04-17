#coding:utf-8  
'''
Created on 2020年1月11日

@author: zfg
'''
import numpy as np       
import matplotlib.pyplot as plt       
from sklearn.cluster import KMeans        
from sklearn.datasets import make_blobs  

# 设置画布的大小
plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
'''
    make_blobs函数是为聚类产生数据集 
    产生一个数据集和相应的标签 
    n_samples:表示数据样本点个数,默认值100 
    n_features:表示数据的维度，默认值是2 
    centers:产生数据的中心点，默认值3 
    shuffle ：洗乱，默认值是True 
    random_state:随机生成器的种子  固定值  
'''
x,y = make_blobs(n_samples=n_samples, random_state=random_state)
print(x,y)

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(x)
# plt画布  子画布
plt.subplot(221)     
plt.scatter(x[:, 0], x[:, 1], c=y_pred)    
plt.title("kmeans01")
plt.savefig("kmeans01.png")


# 将随机产生的数据，做了一些变换,基于变换后的数据进行聚类   玩!!!
transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(x, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("kmeans02")
#
# #
# #
# #
X_filtered = np.vstack((x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)
plt.subplot(223)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("kmeans03")
#
# #
# #
dataMat = []
fr = open("testSet.txt","r")
for line in fr.readlines():
    if line.strip() != "":
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
dataMat = np.array(dataMat)
y_pred = KMeans(n_clusters=4, random_state=random_state).fit_predict(dataMat)
print(y_pred)

plt.subplot(224)
plt.scatter(dataMat[:, 0], dataMat[:, 1], c=y_pred)
plt.title("kmeans04")
plt.show()
