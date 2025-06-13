import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneous transformation matrix s.t. Y ~ H*X
    """
    ##### STUDENT CODE START #####
    A = []
    for i in range(4):
        x, y = X[i, 0], X[i, 1]
        u, v = Y[i, 0], Y[i, 1]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A)
    _, _, VT = np.linalg.svd(A)
    h = VT[-1]
    H = h.reshape(3, 3)
    ##### STUDENT CODE END #####
    return H

if __name__ == "__main__":
    # You could run this file to test out your est_homography implementation
    #   $ python est_homography.py
    # Here is an example to test your code, 
    # but you need to work out the solution H yourself.
    X = np.array([[0, 0], [0, 10], [5, 0], [5, 10]])
    Y = np.array([[3, 4], [4, 11], [8, 5], [9, 12]])
    H = est_homography(X, Y)
    print(H)
    