import numpy as np

class LeastSquareAlgorithm():
    """"to find the least square classifier """

    def __init__(self):
        self.x = None
        self.y = None
        self.parameters = None

    def fit(self, x=None,  y=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.parameters = np.dot(np.dot(np.linalg.inv(np.dot(self.x.T, self.x)), self.x.T), self.y)

    def feature_matrix(self):
        print(self.x)

    def label(self):
        print(self.y)

    def parameters(self):
        return self.parameters

    def predict(self, test=None):
        self.test = np.array(test)
        return np.array(np.dot(self.test, self.parameters) > 0, dtype= 'int')


ls = LeastSquareAlgorithm()

mt = np.array([[1, 5, 6], [3, 7, 5], [6, 2, 8], [3, 8, 5], [4, 8, 6]])
lb = np.array([1, 0, 1, 0, 0])
w = np.array([3, 5, 4])
ls.fit(mt, lb)

print(ls.predict([[1, 5, 6], [3, 7, 5], [6, 2, 8], [3, 8, 5], [4, 8, 6]]))


