# -*- coding: utf-8 -*-

import numpy as np


class reduced_row_echelon_form:
    def __init__(self):
        self.x = None
        self.pivot_num = 0
        self.above_pivot_row_entry_list = []             #this list temporarily stores the entries above current pivot row
        self.is_first_nonzero_entry = True               #to determine the first nonzero entry below current pivot row

    def fit(self, x = None):
        if len(np.array(x).shape) != 2:
            raise ValueError("Matrix should be in 2-dimension form")
        if np.array(x).dtype != 'float' and np.array(x).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.x = np.array(x, dtype='float')
            self.pivot_num = 0
            self.above_pivot_row_entry_list = []  # this list temporarily stores the entries above current pivot row
            self.is_first_nonzero_entry = True     #to determine the first nonzero entry below current pivot row

    def find_rref(self):
        m, n = self.x.shape
        self.x = list(self.x)
        for j in range(n):
            self.above_pivot_row_entry_list = []             #clean the list for each column cycle
            self.is_first_nonzero_entry = True               #renew the boolean value for each column cycle
            for i in range(m):
                if i < self.pivot_num:
                    self.above_pivot_row_entry_list.append(self.x[i][j])
                else:
                    if np.abs(self.x[i][j]) <= 1e-12:
                        continue
                    else:
                        self.x[i] = self.x[i] / self.x[i][j]         #set the first entry to 1
                        if self.is_first_nonzero_entry:     #if the first nonzero entry found, set its row as pivot row
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
    x = [[18,-13,2, -34],[-13,10,0, 33],[2,0,4, 14], [-34, 33, 14, 194]]
    rref.fit(x)
    result = rref.transform()
    print(result)










