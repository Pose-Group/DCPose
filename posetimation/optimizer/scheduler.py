#!/usr/bin/python
# -*- coding:utf8 -*-
import torch.optim
import logging


def build_lr_scheduler(cfg, optimizer, **kwargs):
    logger = logging.getLogger(__name__)
    if cfg.TRAIN.LR_SCHEDULER == 'MultiStepLR':

        last_epoch = kwargs["last_epoch"] if 'last_epoch' in kwargs else -1
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA,
                                                            last_epoch=last_epoch)
        logger.info(
            "=> Use MultiStepLR. MILESTONES : {}. GAMMA : {}. last_epoch : {}".format(cfg.TRAIN.MILESTONES, cfg.TRAIN.GAMMA, last_epoch))
    else:

        logger.error("Please Check if LR_SCHEDULER is valid")
        raise Exception("Please Check if LR_SCHEDULER is valid")

    return lr_scheduler
