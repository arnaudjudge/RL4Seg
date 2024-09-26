import numpy as np
import torch
from vital.data.camus.utils.process import TEDTemporalRegularization
from vital.utils.image.process.morphological import Post2DBigBlob, Post2DFillIntraHoles, Post2DFillInterHoles


def compare_segmentation_with_ae(segmentation_3d, ae_segmentation_3d, dice_thresh=0.9):
    labels = [i for i in set(np.unique(segmentation_3d)) if i != 0]  # don't use background
    label_dice = np.zeros((len(labels)))
    for label in labels:
        sum_seg = np.sum(segmentation_3d == label)
        sum_ae = np.sum(ae_segmentation_3d == label)
        common = np.sum((segmentation_3d == label) & (ae_segmentation_3d == label))
        label_dice = (2 * common) / (sum_seg + sum_ae)
    return label_dice.mean()


# Instantiate the class that handles temporal consistency post-processing over segmentations, using a pretrained
# cardiac AR-VAE from the publicly available models in nathanpainchaud's Comet model registry as a backbone
# NOTE: If you use the provided pretrained AR-VAE, your segmentations should label the left ventricle as 1 and the
#       myocardium as 2.
class MorphologicalAndTemporalCorrectionAEApplicator:
    def __init__(self, pretrained_model_name):
        self.temporal_regularization = TEDTemporalRegularization(autoencoder=pretrained_model_name)

        self.blob = Post2DBigBlob([2, 1])
        self.intra_holes = Post2DFillIntraHoles([2, 1])
        self.inter_holes = Post2DFillInterHoles(1, 2, 2)

    def fix_morphological_and_ae(self, your_image_sequence):
        your_image_sequence = torch.Tensor(your_image_sequence.transpose((2, 1, 0)))

        # morphological operations
        morph_fixed = self.blob(your_image_sequence.cpu().numpy())
        morph_fixed = self.intra_holes(morph_fixed)
        morph_fixed = self.inter_holes(morph_fixed)

        # your_image_sequence: torch.Tensor # Tensor of shape (N, H, W) where N is the temporal dimension
        postprocessed_segmentation, postprocessed_not_reshaped = self.temporal_regularization(morph_fixed)["post_mask"]
        postprocessed_segmentation = postprocessed_segmentation.transpose((2, 1, 0))
        return postprocessed_segmentation, morph_fixed, postprocessed_not_reshaped.transpose((2, 1, 0))
