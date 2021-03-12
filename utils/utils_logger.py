#!/usr/bin/python
# -*- coding:utf8 -*-
import logging
import os
import sys
from sheen import ColoredHandler


# setup root logger
def setup_logger(save_file, logger=None, logger_level=logging.DEBUG, **kwargs):
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(logger_level)
    logger.addHandler(ColoredHandler())

    # log_file
    file_handler = logging.FileHandler(save_file)
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)


def reset_logger(save_file, logger=None, logger_level=logging.DEBUG, **kwargs):
    local_rank = kwargs.get("local_rank", -1)

    if local_rank <= 0:
        if logger is None:
            logger = logging.getLogger()
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])
        setup_logger(save_file, logger, logger_level)
