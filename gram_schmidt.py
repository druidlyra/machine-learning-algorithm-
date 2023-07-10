import numpy as np

class Gram_Schimidt_Base:
    def __init__(self):
        self.mt = None

    def fit(self, mt):
        if len(np.array(mt).shape) != 2:
            raise ValueError("Matrix should be in n * n form")
        if np.array(mt).dtype != 'float' and np.array(mt).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.mt = np.array(mt, dtype='float')

    def base(self):
        n,p = self.mt.shape
        gsb = []
        for i in range(p):                     #stable gram schmidt algorithm
            v = self.mt[:, i]
            if np.abs(np.dot(v, v)) <= 1e-12:
                continue
            else:
                norm = np.sqrt(np.dot(v, v))
                v = v/norm
                gsb.append(v)
            for k in range(len(gsb), p):
                self.mt[:, k] = self.mt[:, k] - np.dot(v, self.mt[:, k]) * v
        return np.array(gsb).T


if __name__ == '__main__':
    i = np.array([[18,-13,2, -34],[-13,10,0, 33],[2,0,4, 14], [-34, 33, 14, 194]])
    GSB = Gram_Schimidt_Base()
    GSB.fit(i)
    result1 = GSB.base()
    print(result1)

