# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    pass

def __init__(self, alpha=1e-12, max_iter=1000000, tol=0.0001):
        self.alpha = alpha  # 正则化系数
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
def lasso(self,data):
    X,y = read_data();
    m,n = X.shape;
    self.theta = np.zeros(n);
    for i in range(self.max_iter):
        #Calculate the gradient
        grad = np.dot(X.T, np.dot(X, self.theta) - y) + self.alpha * np.sign(self.theta)
        #Update theta
        self.theta -= self.alpha * grad
        #Stopping condition
        if np.linalg.norm(grad) < self.tol:
            break
    return np.dot(X,self.theta)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, 
