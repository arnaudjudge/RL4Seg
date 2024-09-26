import numpy as np
import cv2
import torch


def put_text_to_image(img, text):
    return cv2.putText(img.squeeze(0), "{:.3f}".format(text), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (125), 2)


def log_image(logger, img, title, number=0, img_text=None):
    img = img.cpu().numpy()
    if logger is None:
        return
    img = (img.copy() * (255 / max(img.max(), 1))).astype(np.uint8)
    if img_text:
        img = torch.tensor(put_text_to_image(img, img_text)).unsqueeze(0)
    logger.experiment.add_image(title, img, number)
