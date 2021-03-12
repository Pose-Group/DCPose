#!/usr/bin/python
# -*- coding:utf8 -*-
from .mse_loss import JointMSELoss


def build_loss(cfg, **kwargs):
    if cfg.LOSS.NAME == "MSELOSS":
        return JointMSELoss(cfg.LOSS.USE_TARGET_WEIGHT)
