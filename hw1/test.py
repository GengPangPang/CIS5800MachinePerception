import numpy as np
X = np.array([[0, 0], [0, 10], [5, 0], [5, 10]])
Y = np.array([[3, 4], [4, 11], [8, 5], [9, 12]])
d = X[:, 1]
e = d[:, np.newaxis]
print(d)
print(e)