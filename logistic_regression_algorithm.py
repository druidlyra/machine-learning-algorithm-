# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class LogRegClassifier:
    def __init__(self):
        self.x = None
        self.m = None
        self.n = None
        self.w = None
        self.dw = None
        self.b = None
        self.db = None
        self.z = None
        self.dz = None
        self.a = None
        self.da = None
        self.y = None
        self.cost = []
        self.alpha = 2 * 10e-3

    def fit(self, mt, y):
        if len(np.array(mt).shape) != 2:
            raise ValueError("Matrix should be in n * n form")
        if np.array(mt).dtype != 'float' and np.array(mt).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        self.x = mt
        self.y = y
        self.m, self.n = self.x.shape
        self.w = np.random.rand(self.m, 1) * 10e-5
        self.b = 0

    def activation_function(self, z):
        # element_wise step
        a = 1 / (1 + np.exp(-z))
        return a

    def cost_function(self, a, y):
        # element-wise step
        loss = -1 * (y * np.log(a) + (1-y) * np.log(1-a))
        cost = np.sum(loss, axis = 1)/self.n
        return cost

    def forword_propagation(self):
        self.z = np.matmul(self.w.T, self.x) + self.b
        self.a = self.activation_function(self.z)
        cost = self.cost_function(self.a, self.y)
        self.cost.append(cost)

    def back_propagation(self):
        self.da = -1 * y/self.a + (1-y) / (1-self.a)
        self.dz = self.a - y
        self.db = np.sum(self.dz, axis=1) / self.n
        self.dw = np.matmul(self.x, self.dz.T) / self.n

    def optimazation(self):
        self.w = self.w - self.dw * self.alpha
        self.b = self.b - self.db * self.alpha

    def find_parameters(self):
        a=0
        for i in range(200000):
            self.forword_propagation()
            self.back_propagation()
            delta1 = np.sqrt(np.dot(self.w.T, self.w))[0, 0]
            delta2 = self.b
            if delta2 < 10e-6 and delta1 < 10e-6:
                break
            self.optimazation()
            a += 1
        return self.w, self.b, a, self.cost

if __name__ == "__main__":
    iris = load_iris()
    name = iris.target_names
    data = iris.data
    feature = iris.feature_names
    label = iris.target
    mask = label != 0
    label_s = label[mask]
    data_s = data[mask]
    x = np.zeros((data_s.shape[0], 2))
    x[:, 0] = data_s[:, 0]
    x[:, 1] = data_s[:, 2]
    x = x.T
    y = label_s - 1
    y = y.reshape(1, 100)
    lrc = LogRegClassifier()
    lrc.fit(x, y)
    w, b, a, li = lrc.find_parameters()
    x0 = np.linspace(4.5, 8 , 1000)
    x1 = -1 * (w[0] * x0 + b) / w[1]
    fig = plt.figure()
    fig.suptitle('Logistic Regression Classifier - Gredient Descent')
    ax0 = plt.subplot(121)
    ax0.scatter(x[0], x[1], c=label_s)
    ax0.plot(x0, x1, c = 'red')
    ax0.set_xlabel(feature[0])
    ax0.set_ylabel(feature[2])
    ax1 = plt.subplot(122)
    ax1.plot(range(a), li)
    ax1.set_xlabel('Iteration times')
    ax1.set_ylabel('Cost')
    plt.show()










