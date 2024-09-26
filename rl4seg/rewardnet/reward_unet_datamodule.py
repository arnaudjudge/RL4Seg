import random
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader


class RewardNetAutoDataset(Dataset):
    """ Works with the output of 'utils.file_utils.save_batch_to_dataset """
    def __init__(self, data_path, test_frac=0.1, test=False):
        super().__init__()
        self.data_path = data_path
        self.img_list = []

        # only use /images/ folders to get number of individual entries
        for im_file in Path(f"{self.data_path}").rglob("**/images/*.nii.gz"):
            self.img_list += [im_file.as_posix()]

        random.shuffle(self.img_list)

        # split according to test_frac
        test_len = int(test_frac * len(self.img_list))
        if test:
            self.img_list = self.img_list[-test_len:]
        else:
            self.img_list = self.img_list[:-test_len]

        print(f"LEN of reward net dataset: {self.__len__()}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = nib.load(self.img_list[idx]).get_fdata()
        gt = nib.load(self.img_list[idx].replace("images", "gt")).get_fdata()
        pred = nib.load(self.img_list[idx].replace("images", "pred")).get_fdata()
        x = np.vstack((img, pred))

        y = (gt == pred)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class RewardNetDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, data_path, *args, **kwargs):
        super().__init__()
        self.data_path = data_path

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here

        if stage == "fit" or stage is None:
            train_set_full = RewardNetAutoDataset(self.data_path)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = RewardNetAutoDataset(self.data_path, test=True)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=8, num_workers=8)


if __name__ == "__main__":

    dl = RewardNetDataModule('../CGPT_Loop/')
    dl.setup()
    for i in range(1):
       batch = next(iter(dl.train_dataloader()))
       print(batch[0].shape)
       print(batch[1].shape)

       from matplotlib import pyplot as plt
       plt.figure()
       plt.imshow(batch[0][0, 0, :, :].cpu().numpy().T)

       plt.figure()
       plt.imshow(batch[0][0, 1, :, :].cpu().numpy().T)

       plt.figure()
       plt.imshow(batch[1][0, :, :].cpu().numpy().T)
       plt.show()

