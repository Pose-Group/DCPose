#!/usr/bin/python
# -*- coding:utf8 -*-
import logging
import os.path as osp
import time
from utils.utils_folder import create_folder
from utils.utils_logger import reset_logger
from tabulate import tabulate
from termcolor import colored


class BaseExecutor:
    def __init__(self, cfg, output_folders: dict, phase: str, **kwargs):
        self._hooks = []
        self.output_path_dict = {}
        self.cfg = cfg
        self.phase = phase
        self.checkpoints_save_folder = None
        self.tb_save_folder = None
        self.log_file = None

        self.update_output_paths(output_folders, phase)

    def update_output_paths(self, output_paths, phase):
        log_save_folder = output_paths.get("log_save_folder", "./log")
        create_folder(log_save_folder)
        log_file = osp.join(log_save_folder, "{}-{}.log".format(phase, time.strftime("%Y_%m_%d_%H")))

        self.checkpoints_save_folder = output_paths["checkpoints_save_folder"]
        self.tb_save_folder = output_paths["tb_save_folder"]
        self.log_file = log_file

        reset_logger(self.log_file)

        self.show_info()

    def show_info(self):
        logger = logging.getLogger(__name__)
        table_header = ["Key", "Value"]
        table_data = [
            ["Phase", self.phase],
            ["Log File", self.log_file],
            ["Checkpoint Folder", self.checkpoints_save_folder],
            ["Tensorboard_save_folder", self.tb_save_folder],
        ]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Executor Operating Parameter Table: \n" + colored(table, "red"))

    def exec(self):
        raise NotImplementedError

    def __del__(self):
        pass
