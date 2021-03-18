#!/usr/bin/python
# -*- coding:utf8 -*-

from .keypoints_ord import coco2posetrack_ord


def convert_data_to_annorect_struct(poses, tracks, boxes, **kwargs):
    """
            Args:
                boxes (np.ndarray): Nx5 size matrix with boxes on this frame
                poses (list of np.ndarray): N length list with each element as 4x17 array
                tracks (list): N length list with track ID for each box/pose
    """
    num_dets = len(poses)
    annorect = []

    eval_tracking = kwargs.get("eval_tracking", False)
    tracking_threshold = kwargs.get("tracking_threshold", 0)
    for j in range(num_dets):
        score = boxes[j][0, 5]
        if eval_tracking and score > tracking_threshold:
            continue

        point = coco2posetrack_ord(poses[j], global_score=score)  # here poses 是 4*17
        annorect.append({'annopoints': [{'point': point}],
                         'score': [float(score)],
                         'track_id': [tracks[j]]})
    if num_dets == 0:
        annorect.append({
            'annopoints': [{'point': [{
                'id': [0],
                'x': [0],
                'y': [0],
                'score': [-100.0],
            }]}],
            'score': [0],
            'track_id': [0]})
    return annorect
