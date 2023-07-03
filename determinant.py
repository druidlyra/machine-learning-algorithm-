# -*- coding: utf-8 -*-

import numpy as np
import math


class determinant:
    def __init__(self):
        self.x = None
        self.index_list = []

    def fit(self, x):
        if len(np.array(x).shape) != 2 or np.array(x).shape[0] != np.array(x).shape[1]:
            raise ValueError("Matrix should be in n * n form")
        if np.array(x).dtype != 'float' and np.array(x).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.x = np.array(x, dtype='float')

    def permutation_index(self):
        n = self.x.shape[0]
        a = list(range(n))
        for i in range(math.factorial(n) * 1000):   #permutate the index with 1000 * n! times to ensure all the possibilities be documented
            b = list(np.random.permutation(a))
            if b not in self.index_list:
                self.index_list.append(b)

    def permutation_sign(self, index: list):       #determine the permutation sign
        n = self.x.shape[0]
        permutation_time = 0
        if n > 2:
            for i in range(n-1):
                if index[i] == i:
                    continue
                else:
                    for j in range(i+1, n):
                        if index[j] == i:
                            index[i], index[j] = index[j], index[i]
                            permutation_time += 1
                        else:
                            continue
        else:
            if index[0] != 0:
                permutation_time =1
            else:
                permutation_time = 0
        if permutation_time % 2 == 1:
            sign = -1
        else:
            sign = 1
        return sign

    def value(self):
        self.permutation_index()
        n = self.x.shape[0]
        result = 0
        for j in range(len(self.index_list)):
            a = 1
            for i in range(n):
                a = self.x[i][self.index_list[j][i]] * a
            sign = self.permutation_sign(self.index_list[j])
            a = a * sign
            result += a
        return result


if __name__ == '__main__':
    a = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x = [[3,1,-1,5],[ -5,1,3,-4],[ 2,0,3,-1], [1, -5,3,-3]]
    det = determinant()
    det.fit(x)
    result = det.value()
    print(result)




