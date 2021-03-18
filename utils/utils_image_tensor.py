#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import cv2
import numpy as np
from datasets.transforms.build import mean, std


def tensor2im(input_image, imtype=np.uint8):
    """"
        tensor -> numpy , and normalize

    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  #
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        # (BGR)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        # (channels, height, width) to (height, width, channels)

        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)
