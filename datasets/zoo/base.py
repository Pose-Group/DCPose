#!/usr/bin/python
# -*- coding:utf8 -*-

import logging
import numpy as np
from torch.utils.data import Dataset
from .build import get_dataset_name
from tabulate import tabulate
from termcolor import colored
from utils.common import TRAIN_PHASE


class BaseDataset(Dataset):
    def __init__(self, cfg, phase='train', **kwargs):
        self.dataset_name = get_dataset_name(cfg)
        self.phase = phase

        # common init
        self.is_train = True if self.phase == 'train' else False
        self.pixel_std = 200
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.output_dir = cfg.OUTPUT_DIR
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.image_width = self.image_size[0]
        self.image_height = self.image_size[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)

        # normal data augmentation
        self.scale_factor = cfg.TRAIN.SCALE_FACTOR
        self.rotation_factor = cfg.TRAIN.ROT_FACTOR
        self.flip = cfg.TRAIN.FLIP
        self.num_joints_half_body = cfg.TRAIN.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.TRAIN.PROB_HALF_BODY

        # Loss
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT

        self.data = []

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class VideoDataset(BaseDataset):
    """
        A base class representing VideoDataset.
        All other video datasets should subclass it.
    """

    def __init__(self, cfg, phase, **kwargs):
        super(VideoDataset, self).__init__(cfg, phase, **kwargs)

    def __getitem__(self, item):
        raise NotImplementedError

    def show_samples(self):
        logger = logging.getLogger(__name__)
        table_header = ["Dataset_Name", "Num of samples"]
        table_data = [[self.dataset_name, len(self.data)]]

        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Datasets Samples Info : \n" + colored(table, "magenta"))

    def show_data_parameters(self):
        logger = logging.getLogger(__name__)
        table_header = ["Dataset parameters", "Value"]
        table_data = [
            ["BBOX_ENLARGE_FACTOR", self.bbox_enlarge_factor],
            ["NUM_JOINTS", self.num_joints]
        ]
        if self.phase != TRAIN_PHASE:
            table_extend_data = [
                []
            ]
            table_data.extend(table_extend_data)
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Datasets Parameters Info : \n" + colored(table, "magenta"))


class ImageDataset(BaseDataset):
    """
        A base class representing ImageDataset.
        All other image datasets should subclass it.
    """

    def __getitem__(self, item):
        raise NotImplementedError

    def show_samples(self):
        pass
