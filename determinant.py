# -*- coding: utf-8 -*-

import numpy as np


class determinant:
    def __init__(self):
        self.x = None

    def fit(self, x):
        if len(np.array(x).shape) != 2:
            raise ValueError("Matrix should be in 2-dimension form")
        if np.array(x).dtype != 'float' and np.array(x).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.x = np.array(x, dtype='float')

    def value(self):
        n = self.x.shape
        plus_row = self.x[0, :]
        minus_row = self.x[0, :]
        self.x = np.hstack((self.x, self.x))
        for i in range(n[0]):
            if i == 0:
                continue
            else:
                plus_row = plus_row * self.x[i, i:(i+n[0])]
                minus_row = minus_row * self.x[i, (n[0]-i):(2*n[0]-i)]
        result = np.sum(plus_row) - np.sum(minus_row)
        return result


if __name__ == '__main__':
    a = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    det = determinant()
    det.fit(a)
    result = det.value()
    print(result)




