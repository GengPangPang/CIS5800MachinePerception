from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic
    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)
    """
    ##### STUDENT CODE START #####
    # Homography Approach
    # Following slides: Pose from Projective Transformation
    # Step 1: Compute Homography H
    Pw_2D = Pw[:, :2]  # Only take x and y columns from Pw (ignoring z)

    # Step 2: Compute Homography H
    H = est_homography(Pw_2D, Pc) # 顺序不知道

    # Step 3: Compute H' = K^(-1) * H
    H_prime = np.linalg.inv(K) @ H

    # Extract the first two columns of H'
    a = H_prime[:, 0]
    b = H_prime[:, 1]
    c = H_prime[:, 2]  # Translation vector T (unnormalized)

    # Step 4: Perform SVD on [a, b] (3x2 matrix)
    A = np.column_stack((a, b))  # Stack a and b to create a 3x2 matrix
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Step 5: Use U and Vt to compute r1 and r2
    R_12 = U @ Vt  # U is 3x2, Vt is 2x2

    r1 = R_12[:, 0]
    r2 = R_12[:, 1]

    # Step 6: Compute the scale factor λ and normalize
    s1, s2 = S
    λ = (s1 + s2) / 2

    # Ensure orthogonality: compute r3 = r1 x r2
    r3 = np.cross(r1, r2)

    # Step 7: Assemble rotation matrix R
    R = np.column_stack((r1, r2, r3))

    # Step 8: Compute translation vector t
    t = c / λ

    # Step 9: Ensure determinant of R is 1 (make R a valid rotation matrix)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1  # Flip the sign of the third column if det(R) < 0
    R_cw = R.T # 注意老师给的提示
    t_cw = -R_cw @ t
    R = R_cw
    t = t_cw
    ##### STUDENT CODE END #####
    return R, t