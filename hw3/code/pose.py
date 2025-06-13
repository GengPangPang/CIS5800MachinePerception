import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ## Note: each candidate in the above list should be a dictionary with keys "T", "R"
  """ YOUR CODE HERE
  """
  U, _, VT = np.linalg.svd(E)
  Rz_pos90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
  Rz_neg90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
  R0 = U @ Rz_pos90.T @ VT
  R1 = U @ Rz_neg90.T @ VT
  # orthogonal
  if np.linalg.det(R0) < 0:
    R0 = -R0
  if np.linalg.det(R1) < 0:
    R1 = -R1
  T0 = U[:, -1]
  T1 = -U[:, -1]
  transform_candidates = [{"T": T0, "R": R0},
                          {"T": T0, "R": R1},
                          {"T": T1, "R": R0},
                          {"T": T1, "R": R1}]
  """ END YOUR CODE
  """
  return transform_candidates

