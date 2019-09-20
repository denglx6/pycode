import numpy as np
import  math
import matplotlib.pyplot as plt


def f(w, x, m):
    sum = np.zeros(x.shape)
    for i in range(len(x)):
        for j in range(m):
            sum[i] += w[j] * x[i] ** j
    return sum

num_of_point = 15
x_data = np.linspace(0, 1, num_of_point, dtype=np.float32)#生成0到1指定个x坐标
noise = np.random.normal(0, 0.1, x_data.shape).astype(np.float32)#添加均值为0方差为0.1的同维高斯噪声
y_data = np.sin(2 * math.pi * x_data) + noise#生成相应的纵坐标带噪声数据

break_err = 1e-4
max_iteration = 1000
m = 2#表示多项式最高次数加一，即m+1次多项式系数的个数
w = np.random.random(m)#生成输出系数向量
alpha = 0.01#梯度下降学习速率
predict = f(w, x_data, m)
#print(w)
for i in range(max_iteration):
    predict = f(w, x_data, m)
    err = 0.5 * np.sum(np.square(predict - y_data))
    if err <= break_err:
        break
    delta = np.zeros(m)#delta表示梯度，下面求梯度向量
    for j in range(m):
        for k in range(num_of_point):
            delta[j] += (predict[k] - y_data[k]) * (x_data[k] ** j)
    w = w + alpha * delta

#绘图展示
plt.scatter(x_data, y_data)
plt.plot(x_data, predict, 'r--')
plt.show()