#!/usr/bin/python
# -*- coding:utf8 -*-

from utils.utils_registry import MODEL_REGISTRY
from utils.common import TRAIN_PHASE, TEST_PHASE, VAL_PHASE


def build_model(cfg, phase, **kwargs):
    """
        return : model Instance
    """
    model_name = cfg.MODEL.NAME
    model_Class = MODEL_REGISTRY.get(model_name)
    model_instance = model_Class.get_net(cfg=cfg, phase=phase, **kwargs)

    if phase == TRAIN_PHASE and cfg.MODEL.INIT_WEIGHTS:
        model_instance.init_weights()
        model_instance.train()

    if phase != TRAIN_PHASE:
        model_instance.eval()

    return model_instance


def get_model_hyperparameter(args, cfg, **kwargs):
    """
        return : model hyperparameter
    """
    model_name = cfg.MODEL.NAME
    hyper_parameters_setting = MODEL_REGISTRY.get(model_name).get_model_hyper_parameters(args, cfg)
    return hyper_parameters_setting
