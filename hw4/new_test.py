import numpy as np
result = np.ones((4, 5, 2))
result *= 2
result_homogeneous = np.concatenate((result, np.ones((*result.shape[:-1], 1))), axis=-1)

print("Homogeneous result:\n", result_homogeneous)

K = np.array([[3108.427510480831, 0.0, 2035.7826140150432],
              [0.0, 3103.95507309346, 1500.256751469342],
              [0.0, 0.0, 1.0]])

K_inv = np.linalg.inv(K)

print("K_inv", K_inv)

# 使用矩阵乘法对 (N, F, 3) 每个特征点的齐次坐标进行左乘 K_inv
camera_coords = np.einsum('ij,nfj->nfi', K_inv, result_homogeneous)

# 输出结果
print(camera_coords)