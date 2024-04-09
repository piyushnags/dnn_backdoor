from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import Tensor
import torch.nn.functional as F
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pattern_craft(im_size, pattern_type, perturbation_size):
    _, H, W = im_size
    # initialize perturbation tensor
    perturbation = torch.zeros(im_size)
    c = random.randint(0,2)

    if pattern_type == 'pixel':
        # No. of pixels to be perturbed
        N = 4

        # Randomly choose N pixels
        h = [ random.randint(0, H-1) for _ in range(N) ]
        w = [ random.randint(0, W-1) for _ in range(N) ]

        # Add perturbation
        for i in range(N):
            d = random.gauss(1., 0.05)
            x,y = h[i], w[i]
            perturbation[c][x][y] += perturbation_size*d

    elif pattern_type == 'global':
        # Checkerboard pattern
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                d = random.gauss(1., 0.05)
                perturbation[:, i, j] += perturbation_size*d
    
    else:
        raise ValueError(f'{pattern_type} is an invalid pattern type')
    
    return perturbation


def add_backdoor(image, perturbation):
    return torch.clamp(image+perturbation, 0, 1)



class PoisonDataset(Dataset):
    '''
    Wrapper for the Poisoned dataset
    '''
    def __init__(self, dset: Dataset, imgs: List[Tensor],
                 labels: List[Tensor], inds: Tensor):
        super(PoisonDataset, self).__init__()

        if len(inds) > len(imgs):
            raise RuntimeError(f'Number of victim indices exceeds number of available attack images\ninds: {len(inds)}, imgs: {len(imgs)}')

        self.dset = dset
        self.inds = inds
        self.imgs = imgs
        self.labels = labels
    

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], List[Tensor]]:
        # Check if the current index corresponds to a poisoned image
        x = (self.inds==idx).nonzero()[0]
        
        # Not a poisoned image if idx not in inds
        if len(x) == 0:
            return self.dset[idx]
        # Poisoned, return a poisoned image corresponding to the index
        # of stored poisoned indices. 
        else:
            i = x.item()
            return self.imgs[i], self.labels[i].item()

    def __len__(self) -> int:
        return len(self.dset)
