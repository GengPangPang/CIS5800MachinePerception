from lse import least_squares_estimation
import numpy as np

def get_dist(x1, x2, E):
    Ex1 = E @ x1
    e3_hat = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
    dist = (x2.T @ Ex1) ** 2 / np.linalg.norm(e3_hat @ Ex1) ** 2
    return dist

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8
    eps = 10**-4
    best_num_inliers = -1
    best_inliers = None
    best_E = None
    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]
        """ YOUR CODE HERE
        """
        # X1 -> q X2 -> p
        E = least_squares_estimation(X1[sample_indices], X2[sample_indices])
        inliers = list(sample_indices) # sample_indices 也需要放在inliers里面
        for j in test_indices:
            x1 = X1[j]
            x2 = X2[j]
            residual = get_dist(x1, x2, E) + get_dist(x2, x1, E.T)
            if residual < eps:
                inliers.append(j)
        inliers = np.array(inliers)

        """ END YOUR CODE
        """
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers
    return best_E, best_inliers