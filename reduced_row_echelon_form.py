# -*- coding: utf-8 -*-

import numpy as np


class reduced_row_echelon_form:
    def __init__(self):
        self.x = None
        self.pivot_num = 0
        self.above_pivot_row_entry_list = []
        self.is_first_nonzero_entry = True

    def fit(self, x=None):
        if len(np.array(x).shape) != 2:
            raise ValueError("Matrix should be in 2-dimension form")
        if np.array(x).dtype != 'float' and np.array(x).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.x = np.array(x, dtype='float')

    def find_rref(self):
        m, n = self.x.shape
        self.x = list(self.x)
        for j in range(n):
            self.above_pivot_row_entry_list = []  # clean the list for each column cycle
            self.is_first_nonzero_entry = True  # renew the boolean value for each column cycle
            for i in range(m):
                if i < self.pivot_num:
                    self.above_pivot_row_entry_list.append(self.x[i][j])
                else:
                    if np.abs(self.x[i][j]) < 0.00001 :
                        self.x[i][j] = 0
                        continue
                    else:
                        self.x[i] = self.x[i] / self.x[i][j]                      # set the first entry to 1
                        if self.is_first_nonzero_entry:                      # if the first nonzero entry found, set its row as pivot row
                            self.x[self.pivot_num], self.x[i] = self.x[i], self.x[self.pivot_num]
                            for k in range(len(self.above_pivot_row_entry_list)):
                                self.x[k] = self.x[k] - self.x[self.pivot_num] * self.above_pivot_row_entry_list[k]
                            self.pivot_num += 1
                            self.is_first_nonzero_entry = False
                        else:
                            self.x[i] = self.x[i] - self.x[(self.pivot_num - 1)]
        self.x = np.array(self.x)

    def transform(self):
        self.find_rref()
        return self.x


if __name__ == '__main__':
    rref = reduced_row_echelon_form()
    x = [[1, 2, 2, 2], [2, 4, 6, 8], [3, 6, 8, 10]]
    rref.fit(x)
    result = rref.transform()
    print(result)



