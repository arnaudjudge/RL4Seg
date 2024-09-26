from typing import Tuple

import numpy as np
from medpy.metric import dc, hd
from vital.data.config import LabelEnum


def dice(pred: np.ndarray, target: np.ndarray, labels: Tuple[LabelEnum], exclude_bg: bool = True,
         all_classes: bool = False):
    """Compute dice for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the dice.
        exclude_bg: If true, background dice is not considered.

    Returns:
        mean dice
    """
    dices = []
    if len(labels) > 2:
        for label in labels:
            if exclude_bg and label == 0:
                pass
            else:
                pred_mask, gt_mask = np.isin(pred, label), np.isin(target, label)
                dices.append(dc(pred_mask, gt_mask))
        if all_classes:
            dice_dict = {f"dice_{label.name}": dice for label, dice in zip(labels[1:], dices)}
            dice_dict['Dice'] = np.array(dices).mean()
            return dice_dict
        else:
            return np.array(dices).mean()
    else:
        if all_classes:
            return {'Dice': dc(pred.squeeze(), target.squeeze())}
        else:
            return dc(pred.squeeze(), target.squeeze())


def hausdorff(pred: np.ndarray, target: np.ndarray, labels: Tuple[LabelEnum], exclude_bg: bool = True,
         all_classes: bool = False, voxel_spacing: Tuple[float] = None):
    """Compute hausdorff for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the metric.
        exclude_bg: If true, background dice is not considered.

    Returns:
        hausdorff
    """
    hd_dict = {}
    for i in range(len(pred)):
        hausdorffs = []
        for label in labels:
            if exclude_bg and label == 0:
                pass
            else:
                pred_mask, gt_mask = np.isin(pred[i], label), np.isin(target[i], label)
                if pred_mask.sum() == 0:
                    pred_mask[0, 0] = True
                    print('empty mask')
                hausdorffs.append(hd(pred_mask, gt_mask, voxel_spacing[i] if voxel_spacing is not None else None))
        if all_classes:
            for label, haus in zip(labels[1:], hausdorffs):
                hd_dict[f"hd_{label.name}"] = hd_dict.get(f"hd_{label.name}", 0) + haus / len(pred)
            hd_dict['Hausdorff'] = hd_dict.get('Hausdorff', 0) + np.array(hausdorffs).mean() / len(pred)
        else:
            hd_dict['Hausdorff'] = hd_dict.get('Hausdorff', 0) + np.array(hausdorffs).mean() / len(pred)
    return hd_dict
