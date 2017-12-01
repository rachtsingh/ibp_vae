import numpy as np
import torch
import torch.utils.data
import os
import glob
import pdb
from PIL import Image

np.random.seed(3435)

def generate_data(path, N=125, K=10, D=100):
    path = os.path.join(path, "finite_random")
    A = generate_many_features_A((K, D))
    Z = np.random.binomial(1, 0.3, size=(N, K))
    X = np.matmul(Z, A) + np.random.normal(0., 0.1, size=(N, D))
    divide = int(len(X) * 0.8)
    train_X = X[:divide]
    test_X = X[divide:]
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(os.path.join(path, "train.npy"), train_X)
    np.save(os.path.join(path, "test.npy"), test_X)
    np.save(os.path.join(path, "features.npy"), A)

def generate_many_features_A(shape):
    A = np.zeros(shape)
    num_features = shape[0]
    image_size = np.prod(shape[1:])
    for k in range(num_features):
        num = np.random.randint(low=image_size/6 - 3, high=image_size/6 + 3)
        idx = np.random.choice(image_size, num, replace=False)
        A[k].reshape(image_size)[idx] = 1.0
    return A

class FiniteRandom(torch.utils.data.Dataset):
    """
    Generates (if necessary) the finite_random data using 
    """

    def __init__(self, root, mode='train', transform=None, generate=False):
        self.root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform

        if generate or not self._check_exists():
            self.generate()

        if self.mode == 'train':
            path = os.path.join(self.root, "finite_random/train.npy")
            self.train_data = torch.DoubleTensor(np.load(path).astype('float64'))
        elif self.mode == 'test':
            path = os.path.join(self.root, "finite_random/test.npy")
            self.test_data = torch.DoubleTensor(np.load(path).astype('float64'))

    def generate(self):
        if self._check_exists():
            print("finite random dataset exists, skipping generation...")
            return 
        
        generate_data(self.root, 1250, 5, 64)

    def _check_exists(self):
        if self.mode == 'train':
            return os.path.exists(os.path.join(self.root, 'finite_random/train.npy'))
        elif self.mode == 'test':
            return os.path.exists(os.path.join(self.root, 'finite_random/test.npy'))

    def __getitem__(self, index):
        if self.mode == 'train':
            img = self.train_data[index]
        elif self.mode == 'test':
            img = self.test_data[index]

        # if self.transform is not None:
            # pass
        # else:
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            # img = Image.fromarray(img, mode='L')

        # dummy for label
        return img, ()
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'test':
            return len(self.test_data)
