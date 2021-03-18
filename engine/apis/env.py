#!/usr/bin/python
# -*- coding:utf8 -*-

import random
import torch
import numpy as np
import torch.cuda as cuda

__all__ = ['set_random_seed']


def set_random_seed(seed: int):
    if seed == -1:
        seed = random.randint(0, 99999)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)

    return seed
