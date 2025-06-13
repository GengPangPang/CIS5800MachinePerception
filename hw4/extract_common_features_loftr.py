import torch
import cv2
import numpy as np
from functools import reduce
import pycolmap # pycolmap==0.6.1 OK
from loftr import LoFTR, default_cfg

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def extract_features(matcher, image_pair, filter_with_conf=True):
    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    sz_img = (1280, 960)  # ED TA
    img0_raw = cv2.resize(img0_raw, sz_img)
    img1_raw = cv2.resize(img1_raw, sz_img)
    img0 = torch.from_numpy(img0_raw)[None][None].to(device, dtype=torch.float32) / 255.0
    img1 = torch.from_numpy(img1_raw)[None][None].to(device, dtype=torch.float32) / 255.0
    batch = {'image0': img0, 'image1': img1}

    #############################  TODO 4.4 BEGIN  ############################
    # Inference with LoFTR and get prediction
    # The model `matcher` takes a dict with keys `image0` and `image1` as input,
    # and writes fine features back to the same dict.
    # You can get the results with keys:
    #   key         :   value
    #   'mkpts0_f'  :   matching feature coordinates in image0 (N x 2)
    #   'mkpts1_f'  :   matching feature coordinates in image1 (N x 2)
    #   'mconf'     :   confidence of each matching feature    (N x 1)
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        if filter_with_conf:
            mconf = batch['mconf'].cpu().numpy()
            mask = mconf > 0.5  # filter feature with confidence higher than 0.5
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]
    #############################  TODO 4.4 END  ############################
        # 我自己加的测试代码
        # print("mkpts0 content:")
        # print(mkpts0)
        # print("mkpts0 dtype:", mkpts0.dtype)
        # print("mkpts0 shape:", mkpts0.shape)
        #
        # print("matches:", mkpts0.shape[0])
        #
        # print("matches:", mkpts0.shape[0])

        # RANSAC options
        ransac_options = pycolmap.RANSACOptions()
        ransac_options.max_error = 2.0  # Max pixel reprojection error
        ransac_options.confidence = 0.99  # Confidence level
        ransac_options.min_num_trials = 100  # Minimum RANSAC iterations
        ransac_options.max_num_trials = 1000  # Maximum RANSAC iterations
        inliers = pycolmap.fundamental_matrix_estimation(mkpts0, mkpts1, ransac_options)['inliers'] # 3.11.1 version not OK
        print("inliers:", np.count_nonzero(inliers))

        mkpts0 = mkpts0[inliers]
        mkpts1 = mkpts1[inliers]

        return mkpts0, mkpts1

#############################  TODO 4.5 BEGIN  ############################
# Any helper functions you need for this part
# extract_features仅限于处理两张图像之间的特征匹配，而不是在整个数据集中找到所有图像的共同特征
# find_common_features能找到所有图像之间的共同特征
# 代码可能需要优化？运行慢
def find_common_features(image_dir, image_names, matcher, filter_with_conf=True):
    # ref_path = f"{image_dir}/{image_names[0]}"
    ref_path = f"{image_dir}/{image_names[-1]}"
    all_features = []  # Store matching features for each frame

    # Extract features between Frame 0 and every other frame
    # for target_image in image_names[1:]:
    for target_image in image_names[:-1]:
        target_path = f"{image_dir}/{target_image}"
        mkpts0, _ = extract_features(matcher, (ref_path, target_path), filter_with_conf)
        all_features.append(mkpts0)

    # Use reduce to find exact common features
    common_features = reduce(
        lambda x, y: np.array([feat for feat in x if np.any(np.all(feat == y, axis=1))]),
        all_features
    )
    # Build the final N x F x 2 array
    N = len(common_features)
    F = len(image_names)
    result = np.zeros((N, F, 2))
    # Frame 0's features are the reference features
    result[:, 0, :] = common_features
    # Align features for other frames
    for i, target_image in enumerate(image_names[1:], start=1):
        target_path = f"{image_dir}/{target_image}"
        mkpts0, mkpts1 = extract_features(matcher, (ref_path, target_path), filter_with_conf)
        aligned_features = np.array([
            mkpts1[np.argmax(np.all(mkpts0 == feat, axis=1))] for feat in common_features
        ])
        result[:, i, :] = aligned_features

    # sz_img = (1280, 960) # 这种normalize的方法是不对的!!!
    # image_width, image_height = sz_img
    # # Normalize to [-1, 1]
    # result[:, :, 0] = 2 * result[:, :, 0] / image_width - 1  # Normalize x to [-1, 1]
    # result[:, :, 1] = 2 * result[:, :, 1] / image_height - 1  # Normalize y to [-1, 1]
    result *= 3.15 # ED 242
    result_homogeneous = np.concatenate((result, np.ones((*result.shape[:-1], 1))), axis=-1)
    return result_homogeneous

##############################  TODO 4.5 END  #############################

def main():
    # Dataset
    image_dir = './data/pennlogo/'
    image_names = ['IMG_8657.jpg', 'IMG_8658.jpg', 'IMG_8659.jpg', 'IMG_8660.jpg', 'IMG_8661.jpg']
    K = np.array([[3108.427510480831, 0.0, 2035.7826140150432], 
                  [0.0, 3103.95507309346, 1500.256751469342], 
                  [0.0, 0.0, 1.0]])

    # LoFTR model
    matcher = LoFTR(config=default_cfg)
    # Load pretrained weights
    checkpoint = torch.load("weights/outdoor_ds.ckpt", map_location=device, weights_only=True)
    matcher.load_state_dict(checkpoint["state_dict"])
    matcher = matcher.eval().to(device)

    #############################  TODO 4.5 BEGIN  ############################
    # Find common features
    # You can add any helper functions you need
    cf = find_common_features(image_dir, image_names, matcher, filter_with_conf=True) # N x F x 2
    K_inv = np.linalg.inv(K)
    common_features = np.einsum('ij,nfj->nfi', K_inv, cf)
    common_features = common_features[..., :2] / common_features[..., 2:]
    ##############################  TODO 4.5 END  #############################

    np.savez("loftr_frame_features.npz", data=common_features, image_names=image_names, intrinsic=K)

if __name__ == '__main__':
    main()