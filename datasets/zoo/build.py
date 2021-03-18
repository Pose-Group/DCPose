#!/usr/bin/python
# -*- coding:utf8 -*-

import torch.utils.data
from utils.utils_registry import DATASET_REGISTRY

__all__ = ["get_dataset_name", "build_train_loader", "build_eval_loader"]


def get_dataset_name(cfg):
    dataset_name = cfg.DATASET.NAME
    if dataset_name == "PoseTrack":
        dataset_version = "18" if cfg.DATASET.IS_POSETRACK18 else "17"
        dataset_name = dataset_name + dataset_version

    return dataset_name


# TODO Change to dataloader distributed in the future
# for train loader
def build_train_loader(cfg, **kwargs):
    cfg = cfg.clone()
    dataset_name = cfg.DATASET.NAME
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg=cfg, phase='train')

    batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    return train_loader


# for validation / test loader
def build_eval_loader(cfg, phase):
    cfg = cfg.clone()

    dataset_name = cfg.DATASET.NAME
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg=cfg, phase=phase)
    if phase == 'validate':
        batch_size = cfg.VAL.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    elif phase == 'test':
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    else:
        raise BaseException

    eval_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    return eval_loader
