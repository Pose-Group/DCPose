#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np


def PCK(predict, target, label_size=45, sigma=0.2):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than  the value is 1, otherwise it is 0
    :param predict:         x,y,c
    :param target:          3D numpy
    :param label_size:
    :param sigma:
    :return: 0/21, 1/21, ...
    """

    dist = np.linalg.norm(np.subtract(predict, target), axis=1) / label_size
    match = dist <= sigma
    pck = 1.0 * np.sum(match)


    return pck


if __name__ == '__main__':
    a = [[1, 1], [2, 1]]
    b = [[10, 5], [4, 6]]
    a = np.array(a)
    b = np.array(b)
    PCK(a, b)
