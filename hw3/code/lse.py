import numpy as np

def least_squares_estimation(X1, X2):
  """ YOUR CODE HERE
  """
  # X1 -> q X2 -> p
  N = X1.shape[0]
  A = np.zeros((N, 9))
  for i in range(N):
    x1, y1, z1 = X1[i]
    x2, y2, z2 = X2[i]
    A[i] = [x2 * x1, x2 * y1, x2 * z1,
            y2 * x1, y2 * y1, y2 * z1,
            z2 * x1, z2 * y1, z2 * z1]
  _, _, Vt = np.linalg.svd(A)
  E = Vt[-1].reshape(3, 3)
  U, S, Vt = np.linalg.svd(E)
  S = np.diag([1, 1, 0])
  E = U @ S @ Vt
  """ END YOUR CODE
  """
  return E