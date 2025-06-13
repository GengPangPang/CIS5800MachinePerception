import numpy as np

# 替换为你要加载的 .npz 文件路径
file_path = './data/loftr_frame_features.npz'
# x_min: -0.6388936328628554, x_max: 0.5059261204704567
# y_min: -0.4672588332731804, y_max: 0.3528863088340445
# file_path = './data/sift_features.npz'


# 加载 .npz 文件
data = np.load(file_path)

# 查看 .npz 文件中存储的键
print("Keys in the npz file:", data.keys())

# # 逐个查看每个键对应的内容
# for key in data.keys():
#     print(f"Key: {key}, Data shape: {data[key].shape}, Data type: {data[key].dtype}")
#     print(data[key])  # 直接打印数据（如数据较大，建议打印部分）
print(data['data'])
print(data['data'].shape) # (332, 5, 2)
data_array = data['data']

# 计算最后一个维度的最大值和最小值
x_min, x_max = data_array[:, :, 0].min(), data_array[:, :, 0].max()  # 第一个维度 (x)
y_min, y_max = data_array[:, :, 1].min(), data_array[:, :, 1].max()  # 第二个维度 (y)

# 输出结果
print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")