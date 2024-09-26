import torch.nn.functional as F
import torch
import torchvision
# from torchvision import transforms as T
from torchvision.transforms import functional as F_trans
import numbers
import math
import warnings
import random
import numpy as np
import torch.nn as nn

def Confidence_Mask(a, b, thr):
    a1 = a.clone().detach()
    b1 = b.clone().detach()

    Conf_Mask = torch.mul((a1 > thr).float(), (b1 > thr).float()) + torch.mul((a1 < (1 - thr)).float(),
                                                                              (b1 < (1 - thr)).float())

    return Conf_Mask


def target_computing(m1, m2):
    m1 = (m1 > 0.5).float()
    m2 = (m2 > 0.5).float()

    return m1, m2


class DiceLoss_Conf(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_Conf, self).__init__()

    def forward(self, inputs, targets, mask, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        inputs = torch.mul(inputs, mask)
        targets = torch.mul(targets, mask)

        intersection = torch.sum(inputs * targets, dim=(1, 2, 3))
        total = torch.sum(inputs + targets, dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (total + smooth)

        return torch.log(torch.mean(dice))


class BCELoss_Conf(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss_Conf, self).__init__()

    def forward(self, inputs, targets, mask):
        BCE_mean = F.binary_cross_entropy(inputs * mask, targets * mask, reduction='mean')
        return BCE_mean


class Ens_loss(nn.Module):
    def __init__(self, thr=0.85, weight=None, size_average=True):
        super(Ens_loss, self).__init__()

        self.thr = thr
        self.Dice_Conf = DiceLoss_Conf()

        self.BCE1 = BCELoss_Conf()

    def forward(self, inp1, inp2):
        # inp1 = torch.sigmoid(inp1).unsqueeze(1)
        # inp2 = torch.sigmoid(inp2).unsqueeze(1)
        inp1 = (inp1).unsqueeze(1)
        inp2 = (inp2).unsqueeze(1)

        Conf_mask = Confidence_Mask(inp1, inp2, self.thr)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(Conf_mask[0, 0, ...].cpu().numpy().T)
        # plt.show()

        tar1 = inp1.clone().detach()
        tar2 = inp2.clone().detach()

        tar1, tar2 = target_computing(tar1, tar2)

        loss_Dice = self.Dice_Conf(inp1, tar1, Conf_mask)
        loss_BCE = self.BCE1(inp1, tar1, Conf_mask)

        return loss_BCE, Conf_mask



def pad_if_smaller(img, mask, size, fill=0):
    ow = img.shape[1]
    oh = img.shape[2]
    min_size = min(oh, ow)
    if min_size < size:
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        # print(f'padh: {padh}')
        # print(f'padw: {padw}')
        img = F_trans.pad(img, [int(padw / 2), int(padh / 2), int(padw - padw / 2), int(padh - padh / 2)], fill=fill)
        mask = F_trans.pad(mask, [int(padw / 2), int(padh / 2), int(padw - padw / 2), int(padh - padh / 2)], fill=fill)
    return img, mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomApply(object):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.p = p
        self.transforms = transforms

    def __call__(self, img, mask):
        if self.p < torch.rand(1)[0]:
            return img, mask
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomApply_Customized(object):
    def __init__(self, transforms, p):
        super().__init__()
        self.p = p
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            if self.p > torch.rand(1)[0]:
                img, mask = t(img, mask)
        return img, mask


class One_Of(object):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.l = len(transforms)

    def __call__(self, img, mask):
        selected = torch.randint(self.l, (1,))[0]
        img, mask = self.transforms[selected](img, mask)
        return img, mask


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = torch.randint(self.min_size, self.max_size, (1,))[0]
        image = F_trans.resize(image, size)
        target = F_trans.resize(target, size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return image, target


class Resize(object):
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, image, target):
        image = F_trans.resize(image, self.size)
        target = F_trans.resize(target, self.size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return image, target


class Resize_KeepRatio(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F_trans.resize(image, self.size)
        target = F_trans.resize(target, self.size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return image, target


class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, target):
        num_output_channels = F_trans.get_image_num_channels(image)
        if torch.rand(1)[0] < self.p:
            image = F_trans.rgb_to_grayscale(image, num_output_channels=num_output_channels)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if torch.rand(1)[0] < self.flip_prob:
            image = F_trans.hflip(image)
            target = F_trans.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if torch.rand(1)[0] < self.flip_prob:
            image = F_trans.vflip(image)
            target = F_trans.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, self.size[0] - h)
        pad_lr = max(0, self.size[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        h, w = image.shape[1], image.shape[2]
        i = torch.randint(0, h - self.size[0], (1,))[0]
        j = torch.randint(0, w - self.size[1], (1,))[0]
        image = image[:, i:i + self.size[0], j:j + self.size[1]]
        mask = mask[:, i:i + self.size[0], j:j + self.size[1]]
        return image, mask


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F_trans.center_crop(image, self.size)
        target = F_trans.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target.astype(float)).long()
        # target = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F_trans.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RemoveWhitelines(object):
    def __call__(self, image, target):
        target = torch.where(target == 255, 0, target)
        return image, target


class RandomRotation(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = torch.rand(1)[0] * 2 * self.degree - self.degree
        rotate_degree = float(rotate_degree)
        return F_trans.rotate(img, rotate_degree), F_trans.rotate(mask, rotate_degree)


class GaussianBlur(object):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img, mask):
        return F_trans.gaussian_blur(img, self.kernel_size, self.sigma), mask


# My RandomAdjustSharpness is different from PyTorch. My function randomly selects a float number between 1 and the given number as the sharpness factor

class RandomAdjustSharpness(object):
    def __init__(self, sharpness_factor):
        self.sharpness_factor = sharpness_factor
        self.r1 = 1
        self.r2 = sharpness_factor

    def __call__(self, img, mask):
        sharpnessFactor = (self.r1 - self.r2) * torch.rand(1)[0] + self.r2
        return F_trans.adjust_sharpness(img, sharpnessFactor), mask


class ColorJitter(object):
    """Mostly from the docs"""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, img, mask):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F_trans.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F_trans.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F_trans.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F_trans.adjust_hue(img, hue_factor)

        return img, mask


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, img, mask):
        return img + torch.randn(img.size()) * self.std + self.mean, mask

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomResizedCrop(object):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
        super().__init__()

        self.size = size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img):

        width, height = F_trans.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img)
        img = F_trans.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        mask = F_trans.resized_crop(mask, i, j, h, w, self.size, self.interpolation)
        return img, mask

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string


if __name__ == '__main__':
    a = torch.rand((1, 3, 512, 512))
    # RandomRotate(20)(a, a)
    t = Compose([RandomApply([RandomRotation(30), GaussianNoise()]), ColorJitter(brightness=1)])
    t(a, a[:, 0])
