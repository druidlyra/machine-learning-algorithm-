# -*- coding: utf-8 -*-

import numpy as np
from gram_schmidt import Gram_Schimidt_Base
from nullspace import NullSpace

class QR:
    def __init__(self):
        self.mt = None
        self.gsb = None
        self.q_matrix = None
        self.r_matrix = None
        self.ns = None

    def fit(self, mt):
        if len(np.array(mt).shape) != 2:
            raise ValueError("Matrix should be in n * n form")
        if np.array(mt).dtype != 'float' and np.array(mt).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.mt = np.array(mt, dtype='float')
            self.gsb = Gram_Schimidt_Base()
            self.ns = NullSpace()
            self.gsb.fit(mt)
            self.q_matrix = None
            self.r_matrix = None

    def find_q_r(self):
        self.q_matrix = self.gsb.base()
        n = self.mt.shape[0]
        q_matrix_n = self.q_matrix.shape[1]
        if q_matrix_n < n:                                 #extend the
            self.ns.fit(self.q_matrix.T)
            tensor = self.ns.null_space()
            norm = np.sqrt(np.matmul(tensor, tensor.T))
            tensor = tensor/norm
            self.q_matrix = np.hstack((self.q_matrix, tensor.T))
        qt = self.q_matrix.T
        self.r_matrix = np.matmul(qt, self.mt)
        return self.q_matrix, self.r_matrix

if __name__ == '__main__':
    i = np.array([[18,-13,2, -34],[-13,10,0, 33],[2,0,4, 14], [-34, 33, 14, 194]])
    qr= QR()
    qr.fit(i)
    q, r = qr.find_q_r()
    print(q)
    print(r)


