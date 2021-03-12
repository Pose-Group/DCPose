#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
from datasets.zoo.posetrack import PoseTrack_Official_Keypoint_Ordering, PoseTrack_COCO_Keypoint_Ordering


def coco2posetrack_ord(preds, global_score=1):
    # print(xy)
    data = []
    src_kps = PoseTrack_COCO_Keypoint_Ordering
    dst_kps = PoseTrack_Official_Keypoint_Ordering

    global_score = float(global_score)
    dstK = len(dst_kps)
    for k in range(dstK):
        # print(k,dst_kps[k])

        if dst_kps[k] in src_kps:
            ind = src_kps.index(dst_kps[k])
            local_score = (preds[2, ind] + preds[2, ind]) / 2.0
            # conf = global_score
            conf = local_score * global_score
            # if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if True:
                data.append({'id': [k],
                             'x': [float(preds[0, ind])],
                             'y': [float(preds[1, ind])],
                             'score': [conf]})
        elif dst_kps[k] == 'neck':
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')
            x_msho = (preds[0, rsho] + preds[0, lsho]) / 2.0
            y_msho = (preds[1, rsho] + preds[1, lsho]) / 2.0
            local_score = (preds[2, rsho] + preds[2, lsho]) / 2.0
            # conf_msho = global_score
            conf_msho = local_score * global_score

            # if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if True:
                data.append({'id': [k],
                             'x': [float(x_msho)],
                             'y': [float(y_msho)],
                             'score': [conf_msho]})
        elif dst_kps[k] == 'head_top':
            # print(xy)
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')

            x_msho = (preds[0, rsho] + preds[0, lsho]) / 2.0
            y_msho = (preds[1, rsho] + preds[1, lsho]) / 2.0

            nose = src_kps.index('nose')
            x_nose = preds[0, nose]
            y_nose = preds[1, nose]
            x_tophead = x_nose - (x_msho - x_nose)
            y_tophead = y_nose - (y_msho - y_nose)
            local_score = (preds[2, rsho] + preds[2, lsho]) / 2.0
            #
            # if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if True:
                data.append({
                    'id': [k],
                    'x': [float(x_tophead)],
                    'y': [float(y_tophead)],
                    'score': [local_score]})
    return data


def coco2posetrack_ord_infer(pose, global_score=1, output_posetrack_format=False):
    # pose [x,y,c]
    src_kps = PoseTrack_COCO_Keypoint_Ordering
    dst_kps = PoseTrack_Official_Keypoint_Ordering
    if not output_posetrack_format:
        data = np.zeros((len(dst_kps), 3))
    else:
        data = []
    for dst_index, posetrack_keypoint_name in enumerate(dst_kps):
        if posetrack_keypoint_name in src_kps:
            index = src_kps.index(posetrack_keypoint_name)
            local_score = (pose[index, 2] + pose[index, 2]) / 2
            conf = local_score * global_score
            if not output_posetrack_format:
                data[dst_index, :] = pose[index]
                data[dst_index, 2] = conf
            else:
                data.append({'id': [dst_index],
                             'x': [float(pose[index, 0])],
                             'y': [float(pose[index, 1])],
                             'score': [conf]})


        elif posetrack_keypoint_name == 'neck':
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')
            x_msho = (pose[rsho, 0] + pose[lsho, 0]) / 2.0
            y_msho = (pose[rsho, 1] + pose[lsho, 1]) / 2.0
            local_score = (pose[rsho, 2] + pose[lsho, 2]) / 2.0
            conf_msho = local_score * global_score

            # if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if not output_posetrack_format:
                data[dst_index, 0] = float(x_msho)
                data[dst_index, 1] = float(y_msho)
                data[dst_index, 2] = conf_msho
            else:
                data.append({'id': [dst_index],
                             'x': [float(x_msho)],
                             'y': [float(y_msho)],
                             'score': [conf_msho]})

        elif posetrack_keypoint_name == 'head_top':
            # print(xy)
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')

            x_msho = (pose[rsho, 0] + pose[lsho, 0]) / 2.0
            y_msho = (pose[rsho, 1] + pose[lsho, 1]) / 2.0

            nose = src_kps.index('nose')
            x_nose = pose[nose, 0]
            y_nose = pose[nose, 1]
            x_tophead = x_nose - (x_msho - x_nose)
            y_tophead = y_nose - (y_msho - y_nose)
            local_score = (pose[rsho, 2] + pose[lsho, 2]) / 2.0
            if not output_posetrack_format:
                data[dst_index, 0] = float(x_tophead)
                data[dst_index, 1] = float(y_tophead)
                data[dst_index, 2] = local_score
            else:
                data.append({'id': [dst_index],
                             'x': [float(x_tophead)],
                             'y': [float(y_tophead)],
                             'score': [local_score]})

    return data
