import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """
    ##### STUDENT CODE START #####
    N = pixels.shape[0]
    Pw = np.zeros((N, 3))

    for i in range(N):
        pixel_h = np.array([pixels[i, 0], pixels[i, 1], 1])
        X_c = np.linalg.inv(K) @ pixel_h
        s = R_wc @ X_c
        t = -t_wc[2] / s[2]
        P = t * s + t_wc
        Pw[i] = P

    ##### STUDENT CODE END #####
    return Pw
    # wrong
    # N = pixels.shape[0]
    # pixels_h = np.hstack([pixels, np.ones((N, 1))])
    # K_inv = np.linalg.inv(K)
    # X_c = K_inv @ pixels_h.T
    # Pw = (R_wc @ X_c).T + t_wc
    # Pw[:, 2] = 0