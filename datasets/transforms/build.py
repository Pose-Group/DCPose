#!/usr/bin/python
# -*- coding:utf8 -*-

import torch
import torchvision.transforms as T

# RGB
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def build_transforms(cfg, phase):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return transform


def reverse_transforms(batch_tensor: torch.Tensor):
    """
    tensor
    """
    if batch_tensor.shape[1] == 1:  # grayscale to RGB
        batch_tensor = batch_tensor.repeat((1, 3, 1, 1))

    for i in range(len(mean)):
        batch_tensor[:, i, :, :] = batch_tensor[:, i, :, :] * std[i] + mean[i]

    batch_tensor = batch_tensor * 255
    # RGB -> BGR
    RGB_batch_tensor = batch_tensor.split(1, dim=1)
    batch_tensor = torch.cat([RGB_batch_tensor[2], RGB_batch_tensor[1], RGB_batch_tensor[0]], dim=1)
    return batch_tensor
