# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y=read_data()
    #w = (X^T X)^-1 (X^T y)
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w @ data
def lasso(data):
    alpha = 1e-12  # 正则化系数
    max_iter = 100000  # 最大迭代次数
    tol = 0.0001  # 收敛阈值
    X,y = read_data()
    m,n = X.shape
    theta = np.zeros(n)
    for i in range(max_iter):
        #Calculate the gradient
        grad = 1/m * np.dot(X.T, np.dot(X, theta) - y) + alpha * np.sign(theta)
        #Update theta
        theta -= alpha * grad
        #Stopping condition
        if np.linalg.norm(grad) < tol:
            break
    return theta @ data



def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
