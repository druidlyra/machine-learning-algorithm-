# -*- coding: utf-8 -*-

import numpy as np
from eigenvector_eigenvalue import EIGEN


class SVD:
    def __init__(self):
        self.mt = None

    def fit(self, mt):
        if len(np.array(mt).shape) != 2:
            raise ValueError("Matrix should be in 2 dimension")
        if np.array(mt).dtype != 'float' and np.array(mt).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.mt = np.array(mt, dtype='float')
            self.eigen = EIGEN()

    def find_svd(self):
        unique_sigma_u = []
        unique_sigma_v = []
        m, n = self.mt.shape
        k = min(m, n)
        u_matrix = np.zeros((1, m))
        v_matrix = np.zeros((1, n))
        a_at = np.matmul(self.mt, self.mt.T)    #a_at and at_a are real symmetric matrices
        at_a = np.matmul(self.mt.T, self.mt)    #a_at = u sigma^2 u.T and at_a = v sigma^2 v.T
        self.eigen.fit(a_at)
        sigma_squre_m, u_vec = self.eigen.get_eigvalue_eigvector()
        sigma_m = np.sqrt(sigma_squre_m)
        sigma_matrix = np.zeros((m, n))
        for i in range(k):
            sigma_matrix[i][i] = sigma_m[i]
        for i in range(m):
            if sigma_m[i] not in unique_sigma_u:
                u_matrix = np.vstack((u_matrix, u_vec[i]))
                unique_sigma_u.append(sigma_m[i])
        u_matrix = u_matrix[1:, :].T
        self.eigen.fit(at_a)
        sigma_squre_v, v_vec = self.eigen.get_eigvalue_eigvector()
        for i in range(n):
            if sigma_squre_v[i] not in unique_sigma_v:
                v_matrix = np.vstack((v_matrix, v_vec[i]))
                unique_sigma_v.append(sigma_squre_v[i])
        v_matrix = v_matrix[1:, :].T
        for i in range(n):
            av = np.matmul(self.mt, v_matrix[:, i])
            usigma = np.matmul(u_matrix, sigma_matrix[:, i])
            delta = av -usigma
            if np.dot(delta, delta) > 1e-12:
                v_matrix[:, i] = v_matrix[:, i] * -1
        return u_matrix, sigma_matrix, v_matrix.T


if __name__ == '__main__':
    i = np.array([[-1,1,0, 9],[-4,3,0, 8],[1,0,2, 7]])
    svd = SVD()
    svd.fit(i)
    u, s, vt= svd.find_svd()
    print(u)
    print(s)
    print(vt)
    print(i)
    print(np.matmul(np.matmul(u,s),vt))




