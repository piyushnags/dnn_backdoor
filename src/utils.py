import torch
import torch.nn.functional as F
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pattern_craft(im_size, pattern_type, perturbation_size):
    if pattern_type == 'pixel':
        # No. of pixels to be perturbed
        N = 4

        # initialize perturbation tensor
        perturbation = torch.zeros(im_size)

        # Randomly choose N pixels
        H, W = im_size
        h = [ random.randint(0, H-1) for _ in range(N) ]
        w = [ random.randint(0, W-1) for _ in range(N) ]

        # Add perturbation
        for i in range(N):
            d = random.gauss(1., 0.05)
            x,y = h[i], w[i]
            perturbation[x][y] += perturbation_size*d

    elif pattern_type == 'global':
        raise NotImplementedError()
    
    else:
        raise ValueError(f'{pattern_type} is an invalid pattern type')
    
    return perturbation


def add_backdoor(image, perturbation):
    return torch.clamp(image+perturbation, 0, 1)


