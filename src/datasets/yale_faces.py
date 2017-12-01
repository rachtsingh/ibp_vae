import numpy as np
import torch
import torch.utils.data
import os
import glob
from PIL import Image

np.random.seed(3435)

class YaleFacesData(torch.utils.data.Dataset):
    """ Faces 1-31 are train, 32-39 are test """
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        if self.train:
            self.train_data = []
            basedir = os.path.join(self.root, "train")
            for fname in os.listdir(basedir):
                img = np.load(os.path.join(basedir, fname))
                self.train_data.append(img)
        else:
            self.test_data = []
            basedir = os.path.join(self.root, "train")
            for fname in os.listdir(basedir):
                img = np.load(os.path.join(basedir, fname))
                self.test_data.append(img)

    def __getitem__(self, index):
        if self.train:
            img = self.train_data[index]
        else:
            img = self.test_data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        # dummy for label
        return img, ()
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


