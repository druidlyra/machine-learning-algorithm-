# -*- coding: utf-8 -*-

import numpy as np
import reduced_row_echelon_form

class NullSpace:
    def __init__(self):
        self.x = None
        self.r_hat = None
        self.r_hat_nullspace = None
        self.nullspace = None
        self.pivot_num = 0
        self.independent_column_list = []
        self.free_column_list = []
        self.new_order_list = []
        self.rank = 0
        self.rref = None

    def fit(self, x=None):
        if len(np.array(x).shape) != 2:
            raise ValueError("Matrix should be in 2-dimension form")
        if np.array(x).dtype != 'float' and np.array(x).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.x = np.array(x, dtype='float')
            self.rref = reduced_row_echelon_form.reduced_row_echelon_form()
            self.r_hat = None
            self.r_hat_nullspace = None
            self.nullspace = None
            self.pivot_num = 0
            self.independent_column_list = []
            self.free_column_list = []
            self.new_order_list = []
            self.rank = 0

    def find_rref(self):
        self.rref.fit(self.x)
        self.x = self.rref.transform()
        self.r_hat = self.rref.transform()

    def r_hat_rref(self):
        self.find_rref()
        m, n = self.x.shape
        for i in range(m):
            first_non_zero_entry = True
            for j in range(n):
                if self.x[i, j] == 1 and first_non_zero_entry:
                    self.independent_column_list.append(j)
                    first_non_zero_entry = False
                else:
                    continue
        self.rank = len(self.independent_column_list)
        for i in range(n):
            if i not in self.independent_column_list:
                self.free_column_list.append(i)
        self.new_order_list = self.independent_column_list + self.free_column_list
        for i in range(n):
            self.r_hat[:, i] = self.x[:, self.new_order_list[i]]

    def find_r_hat_nullspace(self):
        self.r_hat_rref()
        m, n = self.x.shape
        if m < n:
            free_part = (-1 * self.r_hat[:self.rank, self.rank:]).T
            independent_part = np.identity(self.x.shape[1] - self.rank)
            self.r_hat_nullspace = np.hstack((free_part, independent_part))
        else:
            if self.rank == n:
                self.r_hat_nullspace = 0
            else:
                free_part = (-1 * self.r_hat[:self.rank, self.rank:]).T
                independent_part = np.identity(self.x.shape[1] - self.rank)
                self.r_hat_nullspace = np.hstack((free_part, independent_part))

    def null_space(self):
        self.find_r_hat_nullspace()
        m, n = self.x.shape
        if m < n:
            a, b = self.r_hat_nullspace.shape
            self.nullspace = np.zeros((a, b))
            for i in range(len(self.new_order_list)):
                self.nullspace[:, self.new_order_list[i]] = self.r_hat_nullspace[:, i]
            return self.nullspace
        else:
            if self.rank == n:
                return 0
            else:
                a, b = self.r_hat_nullspace.shape
                self.nullspace = np.zeros((a,b))
                for i in range(len(self.new_order_list)):
                    self.nullspace[:, self.new_order_list[i]] = self.r_hat_nullspace[:, i]
                return self.nullspace


if __name__ == '__main__':
    ns = NullSpace()
    x = np.array([[18,-13,2, -34],[-13,10,0, 33],[2,0,4, 14], [-34, 33, 14, 194]])
    ns.fit(x)
    result = ns.null_space()
    print(result)