#!/usr/bin/python
# -*- coding:utf8 -*-
from random import random
import cv2
from PIL import ImageFont, ImageDraw, Image


def add_bbox_in_image(image, bbox, color=None, label=None, line_thickness=None, multi_language: bool = False):
    """
    :param image
    :param bbox   -  xyxy
    :param color
    :param label
    :param line_thickness
    :param multi_language
    """
    if color is None:
        color = (random() * 255, random() * 255, random() * 255)

    if line_thickness is None:
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    x1, y1, x2, y2 = map(int, bbox)

    corner1 = (x1, y1)
    corner2 = (x2, y2)

    image_with_bbox = cv2.rectangle(image, corner1, corner2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    if label:
        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=font_thickness / 3, thickness=font_thickness)[0]
        text_corner2 = corner1[0] + t_size[0], corner1[1] - t_size[1] - 3
        cv2.rectangle(image, corner1, text_corner2, -1, cv2.LINE_AA)  # filled
        if not multi_language:
            cv2.putText(image, label, (corner1[0], corner1[1] - 2), 0, font_thickness / 3, [225, 255, 255], thickness=font_thickness,
                        lineType=cv2.LINE_AA)
        else:
            font_path = "font/simsun.ttc"
            font = ImageFont.truetype(font_path, 64)
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((corner1[0], corner1[1] - 2), label, font=font, fill=(225, 255, 255))
    return image_with_bbox
