import numpy as np
import torch
import torch.utils.data
import os
import glob
from PIL import Image
from urllib.request import urlretrieve

np.random.seed(3435)

class FixedBinarization(torch.utils.data.Dataset):
    """
    Downloads (if necessary) the data from Hugo Larochelle's website
    code used from IWAE
    """

    def __init__(self, root, mode='train', transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found; try download=True')

        if self.mode == 'train':
            path = os.path.join(self.root, "fixed/train.npy")
            self.train_data = torch.ByteTensor(np.load(path).astype('uint8'))
        elif self.mode == 'test':
            path = os.path.join(self.root, "fixed/test.npy")
            self.test_data = torch.ByteTensor(np.load(path).astype('uint8'))
        elif self.mode == 'valid':
            path = os.path.join(self.root, "fixed/valid.npy")
            self.valid_data = torch.ByteTensor(np.load(path).astype('uint8'))

    def download(self):
        if self._check_exists():
            print("fixed binarization dataset exists, skipping download...")
            return 

        def lines_to_np_array(lines):
            return np.array([[int(i) for i in line.split()] for line in lines])
        
        dataset_dir = os.path.join(self.root, 'fixed')
        
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        subdatasets = ['train', 'valid', 'test']
        for subdataset in subdatasets:
            filename = 'binarized_mnist_{}.amat'.format(subdataset)
            url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(subdataset)
            local_filename = os.path.join(dataset_dir, filename)
            urlretrieve(url, local_filename)

            with open(os.path.join(dataset_dir, filename)) as f:
                lines = f.readlines()
            processed = lines_to_np_array(lines)
            np.save(os.path.join(dataset_dir, subdataset), processed)
            print("saved {}/{}.npy".format(dataset_dir, subdataset))

    def _check_exists(self):
        if self.mode == 'train':
            return os.path.exists(os.path.join(self.root, 'fixed/train.npy'))
        elif self.mode == 'test':
            return os.path.exists(os.path.join(self.root, 'fixed/test.npy'))
        elif self.mode == 'valid':
            return os.path.exists(os.path.join(self.root, 'fixed/valid.npy'))

    def __getitem__(self, index):
        if self.mode == 'train':
            img = self.train_data[index]
        elif self.mode == 'test':
            img = self.test_data[index]
        elif self.mode == 'valid':
            img = self.valid_data[index]

        if self.transform is not None:
            pass
            # img = self.transform(img)
        else:
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img, mode='L')

        # dummy for label
        return img, ()
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'test':
            return len(self.test_data)
        elif self.mode == 'valid':
            return len(self.valid_data)
