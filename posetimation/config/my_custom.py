#!/usr/bin/python
# -*- coding:utf8 -*-

from yacs.config import CfgNode as _CfgNode
import os.path as osp

BASE_KEY = '_BASE_'


class CfgNode(_CfgNode):

    def merge_from_file(self, cfg_filename):
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = osp.expanduser(base_cfg_file)
            else:
                base_cfg_file = osp.join(osp.dirname(cfg_filename), base_cfg_file)
            with open(base_cfg_file, "r") as base_f:
                base_cfg = self.load_cfg(base_f)
            self.merge_from_other_cfg(base_cfg)
            del cfg[BASE_KEY]
        self.merge_from_other_cfg(cfg)
