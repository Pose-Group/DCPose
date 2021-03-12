#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np


def box2cs(box, aspect_ratio, enlarge_factor=1.0):
    """
        box( x y w h ) convert to center and scale

        x,y is top left corner
    """
    x, y, w, h = box[:4]
    return xywh2cs(x, y, w, h, aspect_ratio, enlarge_factor)


def cs2box(center, scale, pixel_std=200, pattern="xywh"):
    """
        center, scale convert to bounding box
        pattern in ["xywh","xyxy"] . default: "xywh"
            xywh - xy upper left corner of bbox , w h is width and height of bbox respectively
            xyxy - upper left corner and bottom right corner
    """
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std

    if pattern == "xyxy":
        # "xyxy" pattern
        x1 = center[0] - w * 0.5
        y1 = center[1] - h * 0.5
        x2 = center[0] + w * 0.5
        y2 = center[1] + h * 0.5
        return [x1, y1, x2, y2]
    else:
        # "xywh" pattern
        x = center[0] - w * 0.5
        y = center[1] - h * 0.5
        return [x, y, w, h]


def xywh2cs(x, y, w, h, aspect_ratio, enlarge_factor):
    center = np.zeros(2, dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    pixel_std = 200
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * enlarge_factor

    return center, scale
