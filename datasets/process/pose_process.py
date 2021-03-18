#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints * joints_vis, joints_vis


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4, \
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def half_body_transform(joints, joints_vis, num_joints, upper_body_ids, aspect_ratio, pixel_std):
    upper_joints = []
    lower_joints = []
    for joint_id in range(num_joints):
        if joints_vis[joint_id][0] > 0:
            if joint_id in upper_body_ids:
                upper_joints.append(joints[joint_id])
            else:
                lower_joints.append(joints[joint_id])

    if np.random.randn() < 0.5 and len(upper_joints) > 2:
        selected_joints = upper_joints
    else:
        selected_joints = lower_joints if len(lower_joints) > 2 else upper_joints

    if len(selected_joints) < 2:
        return None, None

    selected_joints = np.array(selected_joints, dtype=np.float32)
    center = selected_joints.mean(axis=0)[:2]

    left_top = np.amin(selected_joints, axis=0)
    right_bottom = np.amax(selected_joints, axis=0)

    w = right_bottom[0] - left_top[0]
    h = right_bottom[1] - left_top[1]

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)

    scale = scale * 1.5

    return center, scale
