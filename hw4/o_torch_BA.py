import torch
import numpy as np
from batch_rodrigue import batch_rodrigues


class torch_BA(torch.nn.Module):

    def __init__(self, F: int, N: int, features: torch.Tensor, device: str, lr: float) -> None:
        """
        The initial of torch BA module
        steps: optimized steps
        F: represents the target frame number
        N: represents number of features
        features: [N, F, 2] eg (x, y) calibrated feature point x and y in each frame
        """
        super().__init__()
        self.N = N
        self.F = F
        self.features = features.to(device)
        self.device = device
        assert features.shape[0] == N and features.shape[1] == F and features.shape[2] == 2

        #############################  TODO 4.1 BEGIN  ############################
        # DO NOT CHANGE THE NAME, BECAUSE IT FILLS IN THE OPIMIZOR
        self.theta = torch.nn.Parameter((..., device=device), requires_grad=True)
        self.trans = torch.nn.Parameter((..., device=device), requires_grad=True)
        self.key3d = torch.nn.Parameter((..., device=device), requires_grad=True)

        #############################  TODO 4.1 END  ##############################

        self.config_optimizer(lr=lr)

    def forward_one_step(self):
        """
        Forward function running one opimization
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def reprojection(self, training=True):
        """
        Reproject 3d keypoints to each frame
        You can use provided function batch_rodrigues to convert axis angle representation to rotation matrix
        """
        device = self.device
        #############################  TODO 4.2 BEGIN  ############################

        #############################  TODO 4.2 END  ##############################
        if training == False:
            reproj_features = reproj_features.detach().cpu()

        return reproj_features  # Return normalized calibrated keypoint in (N,F,2), with z = 1 ignored

    def config_optimizer(self, lr):
        self.optimizer = torch.optim.Adam([self.theta, self.trans, self.key3d], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5], gamma=0.1)

    def compute_loss(self):
        """
        Error computation function
        The error is defined as the square of distance between reprojected calibrated points (x',y')
            and given calibrated points (x,y)
        """

        #############################  TODO 4.3 BEGIN  ############################

        #############################  TODO 4.3 BEGIN  ############################
        return torch.mean(loss)

    def scheduler_step(self):
        self.scheduler.step()

    def reset_parameters(self):
        self.__init__(F=self.F, N=self.N, features=self.features)

    def save_parameters(self, to_rotm=True):
        theta = self.theta
        full_theta = torch.cat([theta, torch.zeros((1, 3), device=self.device)], dim=0)
        if to_rotm:
            theta = batch_rodrigues(self.theta)
            full_theta = torch.cat([theta, torch.eye(3, device=self.device).unsqueeze(0)], dim=0)

        trans = torch.cat([self.trans, torch.zeros((1, 3), device=self.device)], dim=0)
        return full_theta.detach().cpu(), trans.detach().cpu(), self.key3d.detach().cpu()
