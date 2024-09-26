import torch
from torchmetrics.functional import dice

from vital.models.segmentation.unet import UNet

"""
Reward functions must each have pred, img, gt as input parameters
"""


class Reward:
    @torch.no_grad()
    def __call__(self, pred, imgs, gt, *args, **kwargs):
        raise NotImplementedError


class RewardUnet(Reward):
    def __init__(self, state_dict_path, temp_factor=1):
        self.net = UNet(input_shape=(2, 256, 256), output_shape=(1, 256, 256))
        self.net.load_state_dict(torch.load(state_dict_path))
        self.temp_factor = temp_factor
        if torch.cuda.is_available():
            self.net.cuda()

    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        return torch.sigmoid(self.net(stack)/self.temp_factor).squeeze(1)


class AccuracyMap(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        actions = torch.round(pred)
        assert actions.shape == gt.shape, \
            print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
        simple = (actions == gt).float()
        simple = simple.mean(dim=(1, 2))
        mean_matrix = torch.zeros_like(pred, dtype=torch.float)
        for i in range(len(pred)):
            mean_matrix[i, ...] = simple[i]
        return mean_matrix


class Accuracy(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        actions = torch.round(pred)
        assert actions.shape == gt.shape, \
            print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
        simple = (actions == gt).float()
        return simple.mean(dim=(1, 2, 3), keepdim=True)


class PixelWiseAccuracy(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        actions = torch.round(pred)
        assert actions.shape == gt.shape, \
            print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
        return (actions == gt).float()


class DiceReward(Reward):
    @torch.no_grad()
    def __call__(self, pred, img, gt):
        dice_score = torch.zeros((len(pred), 1, 1, 1)).to(pred.device)
        for i in range(len(dice_score)):
            dice_score[i, ...] = dice(pred[i, ...], gt[i, ...].to(int))
        return dice_score
