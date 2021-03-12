#!/usr/bin/python
# -*- coding:utf8 -*-
import logging
from engine.core import CORE_FUNCTION_REGISTRY
from tabulate import tabulate
from termcolor import colored


class BaseFunction:

    def _print_name_value(self, name_value, full_arch_name):
        logger = logging.getLogger(__name__)
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)

        table_header = ["Model"]
        table_header.extend([name for name in names])
        table_data = [full_arch_name]
        table_data.extend(["{:.4f}".format(value) for value in values])

        table = tabulate([table_data], tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Result Table: \n" + colored(table, "magenta"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def build_core_function(cfg, *args, **kwargs):
    core_function = CORE_FUNCTION_REGISTRY.get(cfg.CORE_FUNCTION)(cfg, *args, **kwargs)

    return core_function
