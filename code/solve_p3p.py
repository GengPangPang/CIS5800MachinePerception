import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    Pc_homo = np.hstack((Pc, np.ones((4, 1))))
    Pc_norm = (np.linalg.inv(K) @ Pc_homo.T).T
    j1 = Pc_norm[0, :] / np.linalg.norm(Pc_norm[0, :])
    j2 = Pc_norm[1, :] / np.linalg.norm(Pc_norm[1, :])
    j3 = Pc_norm[2, :] / np.linalg.norm(Pc_norm[2, :])
    cos_alpha = np.dot(j2, j3)
    cos_beta = np.dot(j1, j3)
    cos_gamma = np.dot(j1, j2)
    a, b, c = np.linalg.norm(Pw[1] - Pw[2]), np.linalg.norm(Pw[0] - Pw[2]), np.linalg.norm(Pw[0] - Pw[1])
    A4 = ((a ** 2 - c ** 2) / b ** 2 - 1) ** 2 - (4 * c ** 2 / b ** 2) * cos_alpha ** 2

    A3 = 4 * ((a ** 2 - c ** 2) / b ** 2 * (1 - (a ** 2 - c ** 2) / b ** 2) * cos_beta -
              (1 - (a ** 2 + c ** 2) / b ** 2) * cos_alpha * cos_gamma + 2 *
              (c ** 2 / b ** 2) * cos_alpha ** 2 * cos_beta)

    A2 = 2 * (((a ** 2 - c ** 2) / b ** 2) ** 2 - 1 + 2 *
              ((a ** 2 - c ** 2) / b ** 2) ** 2 * cos_beta ** 2 + 2 *
              ((b ** 2 - c ** 2) / b ** 2) * cos_alpha ** 2 - 4 *
              ((a ** 2 + c ** 2) / b ** 2) * cos_alpha * cos_beta * cos_gamma + 2 *
              ((b ** 2 - a ** 2) / b ** 2) * cos_gamma ** 2)

    A1 = 4 * (-((a ** 2 - c ** 2) / b ** 2) * (1 + (a ** 2 - c ** 2) / b ** 2) * cos_beta +
              2 * a ** 2 / b ** 2 * cos_gamma ** 2 * cos_beta -
              (1 - (a ** 2 + c ** 2) / b ** 2) * cos_alpha * cos_gamma)

    A0 = (1 + (a ** 2 - c ** 2) / b ** 2) ** 2 - 4 * (a ** 2 / b ** 2) * cos_gamma ** 2
    coeff = [A4, A3, A2, A1, A0]
    V = np.roots(coeff)
    sols = []
    for v in V:
        u = ((-1 + (a ** 2 - c ** 2) / b ** 2) * v ** 2 - (2 * (a ** 2 - c ** 2) / b ** 2) * cos_beta * v + 1 + (a ** 2 - c ** 2) / b ** 2) / (2 * (cos_gamma - v * cos_alpha))
        s1_options = [
            np.sqrt(a ** 2 / (u ** 2 + v ** 2 - 2 * u * v * cos_alpha)),
            np.sqrt(b ** 2 / (1 + v ** 2 - 2 * v * cos_beta)),
            np.sqrt(c ** 2 / (1 + u ** 2 - 2 * u * cos_gamma))
        ]
        s1_positive = [s for s in s1_options if s > 0]
        s1 = np.median(s1_positive)
        s2 = u * s1
        s3 = v * s1
        Pc_3d = np.array([s1 * j1, s2 * j2, s3 * j3])
        R1, t1 = Procrustes(Pc_3d, Pw[:3])

        error = 0
        for i in range(4):
            P_cam = R1.T @ (Pw[i] - t1)
            p_proj = P_cam / P_cam[2]
            error += np.linalg.norm(p_proj[:2] - Pc_norm[i, :2])
        sols.append((R1, t1, error))
    R, t, _ = min(sols, key=lambda x: x[2])
    ##### STUDENT CODE END #####
    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    U, _, Vt = np.linalg.svd(X_centered.T @ Y_centered)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = Y_mean - R @ X_mean
    ##### STUDENT CODE END #####
    return R, t
