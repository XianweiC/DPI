import random

import numpy as np

import torch
from torch.backends import cudnn


def setup_system(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
