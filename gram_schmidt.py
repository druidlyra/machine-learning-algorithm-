import numpy as np

class Gram_Schimidt_Base:
    def __init__(self):
        self.mt = None

    def fit(self, mt):
        if len(np.array(mt).shape) != 2 or np.array(mt).shape[0] != np.array(mt).shape[1]:
            raise ValueError("Matrix should be in n * n form")
        if np.array(mt).dtype != 'float' and np.array(mt).dtype != 'int':
            raise ValueError("Datatype of matrix should be 'int' or 'float'")
        else:
            self.mt = np.array(mt, dtype='float')

    def base(self):
        """return the gram schmidt bases"""
        n,p = self.mt.shape
        gsb = []
        #set the first non zero column as a base direction, and find the base
        first_base = 0
        for i in range(p):
            v = self.mt[:, i]
            if np.abs(np.dot(v, v)) < 0.0005:
                first_base += 1
                continue
            else:
                norm = np.sqrt(np.dot(v, v))
                gsb.append(v / norm)
                break

        #find other bases
        for i in range(first_base+1, p):
            v = self.mt[:, i]
            for j in range(len(gsb)):
                v = v - np.dot(v, np.array(gsb[j])) * np.array(gsb[j])
            if np.abs(np.dot(v, v)) < 0.0005:
                continue
            else:
                norm = np.sqrt(np.dot(v, v))
                v = v / norm
                gsb.append(v)
        return np.round(np.array(gsb).T, 4)

if __name__ == '__main__':
    i = np.array([[1,-1,4],[2,3,-1],[-1,1,0]])
    GSB = Gram_Schimidt_Base()
    GSB.fit(i)
    print(GSB.base())
