import numpy as np

class Gram_Schimidt_Base():
    def __init__(self, mt = None):
        self.mt = np.array(mt)
        if len(self.mt.shape) != 2:
            raise ValueError("Input should be in 2-dimension form")
        if self.mt.dtype != 'float' and self.mt.dtype != 'int':
            raise ValueError("Input should be 'int' or 'float'")

    def base(self):
        """return the gram schmidt bases"""
        n,p = self.mt.shape
        gsb = []
        #set the first non zero column as a base direction, and find the base
        first_base = 0
        for i in range(p):
            v = self.mt[:, i]
            if np.dot(v, v) == 0:
                first_base += 1
                continue
            else:
                norm = np.sqrt(np.dot(v, v))
                gsb.append(v / norm)
                break

        #find other bases
        for i in range(first_base+1, p):
            for j in range(len(gsb)):
                v = self.mt[:, i]
                v = v - np.dot(v, np.array(gsb[j])) * np.array(gsb[j])
            if np.dot(v, v) == 0:
                continue
            else:
                norm = np.sqrt(np.dot(v, v))
                v = v / norm
                gsb.append(v)
        return np.array(gsb).T

if __name__ == '__main__':
    i = np.array([[4,0,0],[3,4,0],[2,0,1]])
    print(Gram_Schimidt_Base(i).base())
