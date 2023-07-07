# -*- coding: utf-8 -*-

import numpy as np
from QR_decomposition import QR
from nullspace import NullSpace


class EIGEN:
    def __init__(self):
        self.mt = None
        self.a_matrix = None
        self.qr_decomposition = None
        self.eigval = []
        self.eigvec = []
        self.ns_list = []
        self.x_matrix = []
        self.ns = None

    def fit(self, mt):
        if len(np.array(mt).shape) != 2 or np.array(mt).shape[0] != np.array(mt).shape[1]:
            raise ValueError("Matrix should be in n * n form")
        if np.array(mt).dtype != 'float' and np.array(mt).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.mt = np.array(mt, dtype='float')
            self.a_matrix = np.array(mt, dtype='float')
            self.qr_decomposition = QR()
            self.ns = NullSpace()

    def clean_matrix(self, mx: np.array, precise = 6):
        n, p = mx.shape
        for i in range(n):
            for j in range(p):
                if np.abs(mx[i][j]) < 1e-12:
                    mx[i][j] = 0
        return mx

    def qr_iteration(self):
        for i in range(10000):
            self.qr_decomposition.fit(self.a_matrix)
            q, r = self.qr_decomposition.find_q_r()
            q = self.clean_matrix(q)
            r = self.clean_matrix(r)
            self.a_matrix = np.matmul(r, q)

    def get_eigvalue_eigvector(self, precise = 3):
        self.qr_iteration()
        self.a_matrix = np.round(self.a_matrix, 3)
        n = self.a_matrix.shape[0]
        for i in range(n):
            self.eigval.append(self.a_matrix[i][i])
            x_matrix = self.mt - np.identity(n) * self.a_matrix[i][i]
            self.ns.fit(x_matrix)
            eigvec = self.ns.null_space()
            self.eigvec.append(eigvec)
        return self.eigval, self.eigvec


if __name__ == '__main__':
    i = np.array([[-1,1,0],[-4,3,0],[1,0,2]])
    ei = EIGEN()
    ei.fit(i)
    eigval, eigvec = ei.get_eigvalue_eigvector()
    print(eigval)
    print(eigvec)





