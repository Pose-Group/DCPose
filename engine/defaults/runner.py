#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import os.path as osp
from .trainer import DefaultTrainer
from .evaluator import DefaultEvaluator
from utils.utils_folder import create_folder
from engine.defaults import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
from posetimation.zoo import get_model_hyperparameter
from datasets import get_dataset_name
from collections import defaultdict


class DefaultRunner:
    def __init__(self, cfg, args, **kwargs):
        self.cfg = cfg
        self.args = args
        # self.PE_Name = args.detector_name
        self.train = args.train
        self.val = args.val
        self.val_from_checkpoint_id = args.val_from_checkpoint
        self.test = args.test
        self.output_path_dict = defaultdict(str)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((map(str, cfg.GPUS)))
        self.gpus = cfg.GPUS
        self.setup_cfg()

    def setup_cfg(self):
        cfg = self.cfg
        hyper_parameters_setting = get_model_hyperparameter(self.args,self.cfg)
        dataset_name = get_dataset_name(self.cfg)
        cfg.defrost()

        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT_NAME, dataset_name, hyper_parameters_setting)
        cfg.freeze()

        checkpoints_save_folder = osp.join(cfg.OUTPUT_DIR, "checkpoints")
        log_save_folder = osp.join(cfg.OUTPUT_DIR, "log")
        writer_save_folder = osp.join(cfg.OUTPUT_DIR, 'tensorboard')
        create_folder(checkpoints_save_folder)
        create_folder(writer_save_folder)
        create_folder(log_save_folder)

        self.output_path_dict["checkpoints_save_folder"] = checkpoints_save_folder
        self.output_path_dict["tb_save_folder"] = writer_save_folder  # tensorboard save
        self.output_path_dict["log_save_folder"] = log_save_folder

        self.cfg = cfg

    def launch(self):
        if self.train:
            trainer = DefaultTrainer(self.cfg, self.output_path_dict, PE_Name=self.args.PE_Name)
            trainer.exec()

        if self.val:
            evaluator = DefaultEvaluator(self.cfg, self.output_path_dict, PE_Name=self.args.PE_Name,
                                         phase=VAL_PHASE, eval_from_checkpoint_id=self.val_from_checkpoint_id)
            evaluator.exec()

        if self.test:
            evaluator = DefaultEvaluator(self.cfg, self.output_path_dict, PE_Name=self.args.PE_Name,
                                         phase=TEST_PHASE)
            evaluator.exec()
