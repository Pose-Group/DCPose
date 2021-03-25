#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import logging

from datasets.process import get_affine_transform
from datasets.transforms import build_transforms
from datasets.process import get_final_preds
from posetimation.zoo import build_model
from posetimation.config import get_cfg, update_config
from utils.utils_bbox import box2cs
from utils.utils_image import read_image, save_image
from utils.common import INFERENCE_PHASE

# Please make sure that root dir is the root directory of the project
root_dir = os.path.abspath('../')


def parse_args():
    parser = argparse.ArgumentParser(description='Inference pose estimation Network')

    parser.add_argument('--cfg', help='experiment configure file name', required=False, type=str,
                        default="./configs/posetimation/DcPose/posetrack17/model_RSN_inference.yaml")
    parser.add_argument('--PE_Name', help='pose estimation model name', required=False, type=str,
                        default='DcPose')
    parser.add_argument('-weight', help='model weight file', required=False, type=str
                        , default='./DcPose_supp_files/pretrained_models/DCPose/PoseTrack17_DCPose.pth')
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    # philly
    args = parser.parse_args()
    args.rootDir = root_dir
    args.cfg = os.path.abspath(os.path.join(args.rootDir, args.cfg))
    args.weight = os.path.abspath(os.path.join(args.rootDir, args.weight))
    return args


cfg = None
args = None


def get_inference_model():
    logger = logging.getLogger(__name__)
    global cfg, args
    args = parse_args()
    cfg = get_cfg(args)
    update_config(cfg, args)
    logger.info("load :{}".format(args.weight))
    checkpoint_dict = torch.load(args.weight)
    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
    new_model = build_model(cfg, INFERENCE_PHASE)
    new_model.load_state_dict(model_state_dict)
    return new_model


model = get_inference_model()
model = model.cuda()
image_transforms = build_transforms(None, INFERENCE_PHASE)
image_size = np.array([288, 384])
aspect_ratio = image_size[0] * 1.0 / image_size[1]


def image_preprocess(image_path: str, prev_image: str, next_image: str, center, scale):
    trans_matrix = get_affine_transform(center, scale, 0, image_size)
    image_data = read_image(image_path)
    image_data = cv2.warpAffine(image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
    image_data = image_transforms(image_data)
    if prev_image is None or next_image is None:
        return image_data
    else:
        prev_image_data = read_image(prev_image)
        prev_image_data = cv2.warpAffine(prev_image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        prev_image_data = image_transforms(prev_image_data)

        next_image_data = read_image(next_image)
        next_image_data = cv2.warpAffine(next_image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        next_image_data = image_transforms(next_image_data)

        return image_data, prev_image_data, next_image_data


def inference_PE(input_image: str, prev_image: str, next_image: str, bbox):
    """
        input_image : input image path
        prev_image : prev image path
        next_image : next image path
        inference pose estimation
    """
    center, scale = box2cs(bbox, aspect_ratio)
    target_image_data, prev_image_data, next_image_data = image_preprocess(input_image, prev_image, next_image, center, scale)

    target_image_data = target_image_data.unsqueeze(0)
    prev_image_data = prev_image_data.unsqueeze(0)
    next_image_data = next_image_data.unsqueeze(0)

    concat_input = torch.cat((target_image_data, prev_image_data, next_image_data), 1).cuda()
    margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).cuda()
    model.eval()

    predictions = model(concat_input, margin=margin)

    pred_joint, pred_conf = get_final_preds(predictions.cpu().detach().numpy(), [center], [scale])
    pred_keypoints = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)

    return pred_keypoints


def inference_PE_batch(input_image_list: list, prev_image_list: list, next_image_list: list, bbox_list: list):
    """
        input_image : input image path
        prev_image : prev image path
        next_image : next image path
        inference pose estimation
    """
    batch_size = len(input_image_list)

    batch_input = []
    batch_margin = []
    batch_center = []
    batch_scale = []
    for batch_index in range(batch_size):
        bbox = bbox_list[batch_index]
        input_image = input_image_list[batch_index]
        prev_image = prev_image_list[batch_index]
        next_image = next_image_list[batch_index]

        center, scale = box2cs(bbox, aspect_ratio)
        batch_center.append(center)
        batch_scale.append(scale)

        target_image_data, prev_image_data, next_image_data = image_preprocess(input_image, prev_image, next_image, center, scale)

        target_image_data = target_image_data.unsqueeze(0)
        prev_image_data = prev_image_data.unsqueeze(0)
        next_image_data = next_image_data.unsqueeze(0)

        one_sample_input = torch.cat((target_image_data, prev_image_data, next_image_data), 1).cuda()
        margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).cuda()

        batch_input.append(one_sample_input)
        batch_margin.append(margin)
    batch_input = torch.cat(batch_input, dim=0).cuda()
    batch_margin = torch.cat(batch_margin, dim=0).cuda()
    # concat_input = torch.cat((target_image_data, prev_image_data, next_image_data), 1).cuda()
    # margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).cuda()
    model.eval()

    predictions = model(batch_input, margin=batch_margin)

    pred_joint, pred_conf = get_final_preds(predictions.cpu().detach().numpy(), batch_center, batch_scale)
    pred_keypoints = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)

    return pred_keypoints
