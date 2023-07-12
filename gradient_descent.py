# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

class GradientDescent:
    def __init__(self):
        self.func = None
        self.w = None
        self.parameter = None
        self.delta = 1e-12
        self.tau = 1e-7
        self.n_iteration = 1000

    def fit(self, f, w, parmeter: tuple, delta = 1e-10, tau = 1e-5, n_iteration = 10000):
        self.func = f
        self.w = np.array(w, dtype = 'float')
        self.parameter = parmeter
        for par in self.parameter:
            par = np.array(par, dtype='float')
        self.delta = delta
        self.tau = tau
        self.n_iteration = n_iteration

    def find_gradient(self, w, p):
        initial = self.func(w, *p)
        df = np.zeros(w.shape[0])
        for i in range(w.shape[0]):
            w_hat = list(w)
            w_hat[i] = w_hat[i] + self.delta
            df[i] = self.func(w_hat, *p) - initial
        gradient = df/self.delta
        return gradient

    def gradient_descent_iteration(self):
        for i in range(self.n_iteration):
            gradient = self.find_gradient(self.w, self.parameter)
            w = self.w - self.tau * gradient
            delta = w -self.w
            if np.abs(np.dot(delta, delta)) < 1e-8:
                print(i)
                break
            self.w = w
        return self.w

if __name__ == '__main__':
    def f(w, x, y, c):
        return np.dot(y - np.matmul(x, w), y - np.matmul(x, w)) + c * np.sqrt(np.dot(w, w))
    x = np.zeros((9, 2))
    x[:, 0] = np.linspace(1, 10, 9) + 1.2* np.sin(np.random.random(9))
    x[:, 1] = np.linspace(1, 10, 9) + 3.3 * np.cos(np.random.random(9))
    y = np.ones(9) - np.random.random(9)- 0.5
    c = 6
    w = np.array([2, 3])
    gd = GradientDescent()
    gd.fit(f, w, (x ,y, c))
    opitimized = gd.gradient_descent_iteration()
    print(opitimized)
    '''
    plt.figure()
    plt.scatter(x[:, 0], y)
    plt.xlim(-1, 12)
    plt.ylim(-1, 12)
    plt.show()
    '''
