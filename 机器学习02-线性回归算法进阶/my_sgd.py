import numpy as np
import matplotlib.pyplot as plt

def h(x):
    return w0 + w1 * x


if __name__ == '__main__':
    # α步长
    rate = 10

    # 准备的样本数据
    # x_train是样本数据的x值
    x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # y_train是样本数据的y值
    y_train = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

    # 随机产生w0
    w0 = np.random.normal()
    # 随机产生w1
    w1 = np.random.normal()
    err = 1


# 能完全等于w0=1 w1=1?

# for循环了10000次  迭代10000次
# for i in range(10000):
# 依据用户指定的误差阈值来收敛
while err > 0.000000001:
    for x, y in zip(x_train, y_train):
        w0 = w0 - rate * (h(x) - y) * 1
        w1 = w1 - rate * (h(x) - y) * x

    # 计算误差值
    err = 0.0
    for x, y in zip(x_train, y_train):
        err += (y - h(x)) ** 2
    m = len(x_train)
    err = float(err / (2 * m))
    print("err:%f" % err)

print(w0, w1)
