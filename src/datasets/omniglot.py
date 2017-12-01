import numpy as np
import pdb
import torch
import torch.utils.data
import os
import glob
import scipy.io
from PIL import Image
from urllib.request import urlretrieve

np.random.seed(3435)

class Omniglot(torch.utils.data.Dataset):
    """
    Omniglot dataset. Code and data from IWAE
    https://github.com/yburda/iwae.git
    """

    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found')

        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28))
        path = os.path.join(self.root, "omniglot.mat")
        omni_raw = scipy.io.loadmat(path)

        if self.train:
            self.train_data = reshape_data(omni_raw['data'].T.astype('float32'))
            # fixed due to seed
            self.train_data = torch.from_numpy(np.random.binomial(1, self.train_data))
        else:
            self.test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
            # fixed due to seed
            self.test_data = torch.from_numpy(np.random.binomial(1, self.test_data))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'omniglot.mat'))

    def __getitem__(self, index):
        if self.train:
            img = self.train_data[index]
        else:
            img = self.test_data[index]

        if self.transform is not None:
            pass
        else:
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img, mode='L')

        # dummy for label
        return img, ()
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
