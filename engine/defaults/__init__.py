#!/usr/bin/python
# -*- coding:utf8 -*-
TRAIN_PHASE = "train"
VAL_PHASE = "validate"
TEST_PHASE = "test"
INFERENCE_PHASE = "inference"
from .argument_parser import default_parse_args
from .runner import DefaultRunner
from .trainer import DefaultTrainer
