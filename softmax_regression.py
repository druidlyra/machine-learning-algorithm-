# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class SoftmaxRegression:
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
        self.y = np.zeros((np.unique(y).shape[0], y.shape[0]))
        for i in range(y.shape[0]):
            self.y[y[i], i] = 1
        self.m, self.n = self.x.shape
        self.w = np.random.rand(self.m, self.y.shape[0]) * 10e-5
        self.b = np.zeros(self.y.shape[0], )

    def activation_function(self, z):
        # element_wise step
        a_prime = np.exp(z)
        sum = np.sum(a_prime, axis = 0)
        a = a_prime/sum
        return a

    def cost_function(self, a, y):
        # element-wise step
        loss_prime = -1 * y * np.log(a)
        loss = np.sum(loss_prime, axis = 0)
        cost = np.sum(loss)/self.n
        return cost

    def forward_propagation(self):
        self.z = np.matmul(self.w.T, self.x) + self.b.reshape(3,1)
        self.a = self.activation_function(self.z)
        cost = self.cost_function(self.a, self.y)
        self.cost.append(cost)

    def back_propagation(self):
        self.da = -1 * self.y / self.a
        self.dz = self.a - self.y
        self.db = np.sum(self.dz, axis=1) / self.n
        self.dw = np.matmul(self.x, self.dz.T) / self.n

    def optimazation(self):
        self.w = self.w - self.dw * self.alpha
        self.b = self.b - self.db * self.alpha

    def find_parameters(self):
        for i in range(20000):
            self.forward_propagation()
            self.back_propagation()
            self.optimazation()
        return self.w, self.b, self.cost

    def predict(self):
        y_hat = np.argmax(self.a, axis=0)
        return y_hat

if __name__ == "__main__":
    soft = SoftmaxRegression()
    iris = load_iris()
    name = iris.target_names
    data = iris.data
    feature = iris.feature_names
    target = iris.target
    x = np.zeros((data.shape[0], 2))
    x[:, 0] = data[:, 0]
    x[:, 1] = data[:, 2]
    x = x.T
    y = target
    soft.fit(x, y)
    w, b, cost = soft.find_parameters()
    y_hat = soft.predict()
    mask = target != y_hat
    fig = plt.figure()
    fig.suptitle('Softmax Regression - Gradient Descent')
    ax0 = plt.subplot(121)
    ax0.scatter(x[0], x[1], c=target)
    ax0.scatter(x[0][mask], x[1][mask], marker = 1, c = 'black')
    ax0.set_xlabel(feature[0])
    ax0.set_ylabel(feature[2])
    ax1 = plt.subplot(122)
    ax1.plot(range(20000), cost)
    ax1.set_xlabel('Iteration times')
    ax1.set_ylabel('Cost')
    plt.show()









