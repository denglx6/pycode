import numpy as np
import  math
import matplotlib.pyplot as plt

def f(w, x, m):
    sum = np.zeros(x.shape)
    for i in range(m):
        sum += w[i] * x ** i
    return sum


x_data = np.linspace(0, 1, 15, dtype=np.float32)
noise = np.random.normal(0, 0.1, x_data.shape).astype(np.float32)
y_data = np.sin(2 * math.pi * x_data) + noise

break_err = 1e-4
max_iteration = 1000
m = 2
w = np.random.random((1, m))
#print(w)
for i in range(max_iteration):
    predict = f(w, x_data, m)
    err = 0.5 * np.sum(np.square(predict - y_data))
    if err <= break_err:
        break
    

#print(y_data)

#plt.scatter(x_data, y_data)
#plt.show()
