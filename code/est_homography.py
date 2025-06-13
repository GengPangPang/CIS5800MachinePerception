import numpy as np

def est_homography(X, Y):
    """ 
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out 
    what X and Y should be. 
    Input:
        X: 4x2 matrix of (x,y) coordinates 
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X
    """
    ##### STUDENT CODE START #####
    # Copy your HW1 code here
    A = []
    for i in range(4):
        x, y = X[i, 0], X[i, 1]
        u, v = Y[i, 0], Y[i, 1]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.array(A)
    _, _, VT = np.linalg.svd(A)
    h = VT[-1]
    H = h.reshape(3, 3)
    H = H / H[-1, -1] # necessary
    ##### STUDENT CODE END #####
    return H
