from random import choice
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False#这两行用于作图时显示中文

data_set = np.random.random((2000, 2))#生成2000个坐标值0到1的二维随机点
def error(point, data):#输入目标点以及数据点集，输出总误差
    total_sum = 0
    for i in range(2000):
        sum = (point[0] - data[i][0]) ** 2 + (point[1] - data[i][1]) ** 2
        total_sum += sum ** 0.5
    return total_sum
def GD(data, alpha, max_iteration, break_err):#data数据点集；alpha学习速率；max_iteration最大迭代步数；break_err终止误差
    plt.figure()
    target_point_list = np.zeros((max_iteration, 2))#记录点迹
    target_point = np.ones(2)#生成初始点
    last_err = 0
    last_iteration = 0#记录最后迭代步数
    for k in range(max_iteration):#迭代求最优点
        last_iteration = k
        target_point_list[k][0] = target_point[0]
        target_point_list[k][1] = target_point[1]#记录点
        err = error(target_point, data)
        if k % 2 == 0:
            print(err)
        if abs(last_err - err) <= break_err:
            print(k)
            break
        last_err = err
        delta = np.zeros(2)#梯度，下面求梯度向量
        for j in range(2000):
            delta[0] += (target_point[0] - data[j][0]) / ((target_point[0] - data[j][0]) ** 2 + (target_point[1] - data[j][1]) ** 2) ** 0.5
            delta[1] += (target_point[1] - data[j][1]) / ((target_point[0] - data[j][0]) ** 2 + (target_point[1] - data[j][1]) ** 2) ** 0.5
        target_point = target_point - alpha * delta#负梯度下降
    print(target_point)
    plt.scatter(target_point_list[0:last_iteration, 0], target_point_list[0:last_iteration, 1])
    plt.plot(target_point_list[0:last_iteration, 0], target_point_list[0:last_iteration, 1], 'r-')
    plt.plot(data[:, 0], data[:, 1], 'g.')

def SGD(data, alpha, max_iteration, break_err):#data数据点集；alpha学习速率；max_iteration最大迭代步数；break_err终止误差
    plt.figure()
    target_point_list = np.zeros((max_iteration, 2))#记录点迹
    target_point = np.ones(2)#生成初始点
    last_err = 0
    last_iteration = 0#记录最后迭代步数
    for k in range(max_iteration):#迭代求最优点
        last_iteration = k
        target_point_list[k][0] = target_point[0]
        target_point_list[k][1] = target_point[1]#记录点
        err = error(target_point, data)
        if k % 50 == 0:
            print(err)
        if abs(last_err - err) <= break_err:
            print(k)
            break
        last_err = err
        delta = np.zeros(2)#梯度，下面求随机梯度向量
        sample = choice(data)#随机选取一个样本点
        #print(sample)
        delta[0] += (target_point[0] - sample[0]) / ((target_point[0] - sample[0]) ** 2 + (target_point[1] - sample[1]) ** 2) ** 0.5
        delta[1] += (target_point[1] - sample[1]) / ((target_point[0] - sample[0]) ** 2 + (target_point[1] - sample[1]) ** 2) ** 0.5
        target_point = target_point - alpha * delta#负梯度下降
    print(target_point)
    plt.scatter(target_point_list[0:last_iteration, 0], target_point_list[0:last_iteration, 1])
    plt.plot(target_point_list[0:last_iteration, 0], target_point_list[0:last_iteration, 1], 'r-')
    plt.plot(data[:, 0], data[:, 1], 'g.')

GD(data_set, 0.0001, 20000, 1e-6)
plt.show()
SGD(data_set, 0.0001, 20000, 1e-6)
plt.show()

