# coding=utf-8
import numpy as np
import  math
import matplotlib.pyplot as plt
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_of_point = 15
x_data = np.linspace(0, 1, num_of_point, dtype=np.float32)#生成0到1指定个x坐标
noise = np.random.normal(0, 0.1, x_data.shape).astype(np.float32)#添加均值为0方差为0.1的同维高斯噪声
y_data = np.sin(2 * math.pi * x_data) + noise#生成相应的纵坐标带噪声数据

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False#这两行用于作图时显示中文

def regressing(x, y, m, break_err=0.2, max_iteration=20000, alpha=0.05):
    def f(w, x, m):
        sum = np.zeros(x.shape)
        for i in range(len(x)):
            for j in range(m):
                sum[i] += w[j] * x[i] ** j
        return sum

    #x表示输入点横坐标向量；y表示输入点纵坐标带噪声向量；break_err表示循环终止误差；max_iteration表示最大迭代数；m表示多项式最高次数加一，即m+1次多项式系数的个数；alpha表示梯度下降学习速率；
    w = np.random.random(m)#初始化多项式系数向量
    predict = f(w, x_data, m)
    for i in range(max_iteration):
        predict = f(w, x_data, m)
        err = 0.5 * np.sum(np.square(predict - y_data))
        if err <= break_err:
            print(i)
            break
        delta = np.zeros(m)#delta表示梯度，下面求梯度向量
        for j in range(m):
            for k in range(num_of_point):
                delta[j] += (predict[k] - y_data[k]) * (x_data[k] ** j)
        w = w - alpha * delta#负梯度下降

    print(w)
    #绘图展示
    new_x = np.linspace(0, 1, 100, dtype=np.float32)
    plt.plot(new_x, np.sin(2 * math.pi * new_x), 'g-', label='Actual line')
    plt.scatter(x_data, y_data, facecolors='none', edgecolors='b')
    plt.plot(new_x, f(w, new_x, m), 'r-', label='Regressing line')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.show()


def regressing_normalized(x, y, m, lamda, break_err=0.2, max_iteration=20000, alpha=0.05):
    def f(w, x, m):
        sum = np.zeros(x.shape)
        for i in range(len(x)):
            for j in range(m):
                sum[i] += w[j] * x[i] ** j
        return sum

    #x表示输入点横坐标向量；y表示输入点纵坐标带噪声向量；break_err表示循环终止误差；max_iteration表示最大迭代数；m表示多项式最高次数加一，即m+1次多项式系数的个数；alpha表示梯度下降学习速率；lamda表示正则项的系数
    w = np.random.random(m)#初始化多项式系数向量
    predict = f(w, x_data, m)
    for i in range(max_iteration):
        predict = f(w, x_data, m)
        err = 0.5 * np.sum(np.square(predict - y_data)) + 0.5 * lamda * np.dot(w, w)
        if(i % 100 == 0):
            print(err)
        if err <= break_err:
            print(i)
            break
        delta = np.zeros(m)#delta表示梯度，下面求梯度向量
        for j in range(m):
            for k in range(num_of_point):
                delta[j] += (predict[k] - y_data[k]) * (x_data[k] ** j) + lamda * w[j]
        w = w - alpha * delta#负梯度下降

    print(w)
    #绘图展示
    new_x = np.linspace(0, 1, 100, dtype=np.float32)
    plt.plot(new_x, np.sin(2 * math.pi * new_x), 'g-', label='Actual line')
    plt.scatter(x_data, y_data, facecolors='none', edgecolors='b')
    plt.plot(new_x, f(w, new_x, m), 'r-', label='Regressing line')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.show()
# regressing(x_data, y_data, 3)
# plt.title('2次多项式回归示意图')
# plt.show()
# regressing(x_data, y_data, 4)
# plt.title('3次多项式回归示意图')
# plt.show()
# regressing(x_data, y_data, 7)
# plt.title('6次多项式回归示意图')
# plt.show()
# regressing(x_data, y_data, 10)
# plt.title('9次多项式回归示意图')
# plt.show()   
regressing_normalized(x_data, y_data, 10, 1e-18, break_err=0.01, max_iteration=500000, alpha=0.05)
plt.title('9次多项式回归示意图')
plt.show()
