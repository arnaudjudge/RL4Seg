import torch
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.metrics.train.functional import differentiable_dice_score


def accuracy(pred, imgs, gt):
    actions = torch.round(pred)
    assert actions.shape == gt.shape, \
        print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
    simple = (actions == gt).float()
    return simple.mean(dim=(1, 2), keepdim=True)


def dice_score(output, target):
    classes = target.unique()
    out = torch.zeros(len(output))
    for i in range(len(output)):
        d = 0
        for c in classes:
            target_c = (target[i] == c)
            output_c = (output[i] == c)
            intersection = torch.sum(target_c * output_c)
            d += (2. * intersection) / (torch.sum(target_c) + torch.sum(output_c))
        out[i] = d / len(classes)
    return out


def is_anatomically_valid(output):
    out = torch.zeros(len(output))
    for i in range(len(output)):
        try:
            out[i] = int(check_segmentation_validity(output[i].T, (1.0, 1.0), [0, 1, 2]))
        except:
            out[i] = 0
    return out

