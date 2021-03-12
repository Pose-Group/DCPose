#!/usr/bin/python
# -*- coding:utf8 -*-
import torch.nn as nn


class BaseModel(nn.Module):

    def forward(self, *input):
        raise NotImplementedError

    def init_weights(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_model_hyper_parameters(cls, cfg):
        """
            get model hyper-parameters
        """
        raise NotImplementedError

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        """
            get model instance
        """
        raise NotImplementedError
