from typing import List

import hydra
from pytorch_lightning import Callback
from omegaconf import DictConfig


# from pl-hydra template (https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/instantiators.py)
def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        # log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            # log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
