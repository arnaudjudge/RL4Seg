import numpy as np
from rl4seg.utils.corrector_utils import MorphologicalAndTemporalCorrectionAEApplicator
from rl4seg.utils.corrector_utils import compare_segmentation_with_ae
from vital.metrics.camus.anatomical.utils import check_segmentation_validity


class Corrector:
    def correct_batch(self, b_img, b_act):
        """
        Correct a batch of images and their actions with AutoEncoder
        Args:
            b_img: batch of images
            b_act: batch of actions from policy

        Returns:
            Tuple (corrected actions, validity of these corrected versions, difference between original and corrected)
        """
        raise NotImplementedError


class AEMorphoCorrector(Corrector):
    def __init__(self, ae_ckpt_path):
        self.ae_corrector = MorphologicalAndTemporalCorrectionAEApplicator(ae_ckpt_path)

    def correct_batch(self, b_img, b_act):
        corrected = np.empty_like(b_img.cpu().numpy())
        corrected_validity = np.empty(len(b_img))
        ae_comp = np.empty(len(b_img))
        for i, act in enumerate(b_act):
            c, _, _ = self.ae_corrector.fix_morphological_and_ae(act.unsqueeze(-1).cpu().numpy())
            corrected[i] = c.transpose((2, 0, 1))

            try:
                corrected_validity[i] = check_segmentation_validity(corrected[i, 0, ...].T, (1.0, 1.0), [0, 1, 2])
            except:
                corrected_validity[i] = False
            ae_comp[i] = compare_segmentation_with_ae(act.unsqueeze(0).cpu().numpy(), corrected[i])
        return corrected, corrected_validity, ae_comp

