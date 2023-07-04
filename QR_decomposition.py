# -*- coding: utf-8 -*-

import numpy as np
from gram_schmidt import Gram_Schimidt_Base

class QR:
    def __init__(self):
        self.mt = None
        self.gsb = None
        self.q_matrix = None
        self.r_matrix = None


    def fit(self, mt):
        if len(np.array(mt).shape) != 2 or np.array(mt).shape[0] != np.array(mt).shape[1]:
            raise ValueError("Matrix should be in n * n form")
        if np.array(mt).dtype != 'float' and np.array(mt).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.mt = np.array(mt, dtype='float')
            self.gsb = Gram_Schimidt_Base()
            self.gsb.fit(mt)

    def find_q_r(self):
        self.q_matrix = self.gsb.base()
        qt = self.q_matrix.T
        self.r_matrix = np.matmul(qt, self.mt)
        return np.round(self.q_matrix, 4), np.round(self.r_matrix, 4)


if __name__ == '__main__':
    i = np.array([[3,1,-1,5],[ -5,1,3,-4],[ 2,0,3,-1], [1, -5,3,-3]])
    qr= QR()
    qr.fit(i)
    q, r = qr.find_q_r()
    print(np.round(np.matmul(q, r), 2))


