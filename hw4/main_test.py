import torch
import tqdm
from visualization import plot_all_poses
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# Set up config for optimization
args = {
    'gpu_id':       '0',
    'data_dir':     './data',
    'result_dir':   './exp',
    'save_dir':     'model',
    'train':        True,
    'lr':           0.001,
    'epoch':        50000
}
os.makedirs(args['save_dir'], exist_ok=True)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:%s" % args['gpu_id'] if use_cuda else "cpu")
dtype = torch.float32
def load_features(path):
    with np.load(path) as data:
        features = data['data']
        frames = data['image_names']
        intri = data['intrinsic']
    return torch.from_numpy(features).to(dtype), frames, torch.from_numpy(intri).to(dtype)
features, frames, K = load_features(os.path.join(args['data_dir'],'loftr_features.npz'))


def visualize_reprojection(images: list, calib_feature: torch.Tensor, reproj: torch.Tensor, K: torch.Tensor):
    """
    images: [np.array]                  Images
    features_2d: torch.tensor[N,F,2]    Given calibrated features
    reproj: torch.tensor[N,F,2]         Reprojected 3d features normalized by z
    K: torch.tensor[3,3]                Intrinsics in
    """

    def p2(calibrated_points):
        """
        Calibrated points to pixels
        """
        key3d = torch.cat(
            [calibrated_points, torch.ones((calibrated_points.shape[0], calibrated_points.shape[1], 1)).to(dtype)],
            dim=-1) @ K.T
        key2d = key3d[:, :, :2] / key3d[:, :, 2].unsqueeze(-1)
        # # 我自己加的测试代码
        # # Print per-dimension min and max values
        # print("Key2d (x) min:", torch.min(key2d[:, :, 0]))
        # print("Key2d (x) max:", torch.max(key2d[:, :, 0]))
        # print("Key2d (y) min:", torch.min(key2d[:, :, 1]))
        # print("Key2d (y) max:", torch.max(key2d[:, :, 1]))
        key2d[:, :, 0] = key2d[:, :, 0].clamp(0, 4032)
        key2d[:, :, 1] = key2d[:, :, 1].clamp(0, 3024)
        return key2d

    if calib_feature is not None:
        calib_2d = p2(calib_feature)

    if reproj is not None:
        reproj_2d = p2(reproj)

    fig, ax = plt.subplots(1, len(images), figsize=(20, 5))
    for f in range(len(images)):
        ax[f].imshow(images[f])
        if calib_feature is not None:
            ax[f].plot(calib_2d[:, f, 0], calib_2d[:, f, 1], 'ro', markersize=2)
        if reproj is not None:
            ax[f].plot(reproj_2d[:, f, 0], reproj_2d[:, f, 1], 'bo', markersize=2)
    plt.show()


# Load features and frames
images = []
for f in range(frames.shape[0]):
    image = cv2.imread(os.path.join('./data/pennlogo', frames[f]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

# Test with given calibrated features
# visualize_reprojection(images, features, None, K)
# visualize_reprojection(images, None, features, K)
# Prepare data for BA
batch = {
        'features': features.to(torch.float32),
        'N': features.shape[0],
        'F': len(frames)
    }

# Once you finished torch_BA.py you can run this training scrip to optimize parameters.
from torch_BA import torch_BA
min_loss = torch.inf
epoch = args['epoch']

BA_backend = torch_BA(batch['F'], batch['N'], batch['features'], device, args['lr'])
loss_values = [] # 自己加的

if args['train']:
    pbar = tqdm.tqdm(range(epoch))
    for iter in pbar:
        loss = BA_backend.forward_one_step()
        loss_values.append(loss)
        pbar.set_postfix({
            'loss': float(loss)
        })
        if iter % 2000 == 0:
            key3d_norm = BA_backend.reprojection(False)
            # viz = visualize_reprojection(images, features, key3d_norm, K)

            if loss< min_loss:
                torch.save({
                    'model_state_dict': BA_backend.state_dict(),
                }, '%s/model.pkl' % args['save_dir'])
else:
    checkpoint = torch.load('%s/model.pkl' % args['save_dir'], map_location=device)
    BA_backend.load_state_dict(checkpoint['model_state_dict'])
# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('LoFTR Loss Curve During Training')
plt.legend()
plt.grid(True)
plt.show()
print(loss_values)
# After optimization you can copy all the code to a .py file and plot the figure with interactive window.
# Set args['Train']=False

theta, trans, key3d = BA_backend.save_parameters(to_rotm=True)
theta = torch.inverse(theta)
trans = (theta @ -trans.unsqueeze(-1) ).squeeze(-1)
poses = torch.cat([theta, trans.unsqueeze(-1)], dim=-1)
plot_all_poses(poses.numpy(), key3d.numpy())

# for colmap parameters, save the parameters for gradescope evaluation
# SUBMIT THIS TO GRADESCOPE
theta, trans, key3d = BA_backend.save_parameters(to_rotm=False)
# np.savez("colmap_parameters.npz", theta = theta.numpy(), trans = trans.numpy(), key3d = key3d.numpy())