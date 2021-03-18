#!/usr/bin/python
# -*- coding:utf8 -*-
import os.path as osp
import argparse


def default_parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', type=str, required=True)
    parser.add_argument('--PE_Name', type=str, default='DcPose')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--val_from_checkpoint',
                        help='exec val from the checkpoint_id. if config.yaml specifies a model file, this parameter will invalid',
                        type=int,
                        default='-1')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--root_dir', type=str, default='../')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.rootDir = osp.abspath(args.root_dir)
    args.cfg = osp.join(args.rootDir, osp.abspath(args.cfg))
    args.PE_Name = args.PE_Name.upper()
    return args
