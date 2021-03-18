#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import torch.nn
import logging
from tensorboardX import SummaryWriter

from .base import BaseExecutor
from .checkpoints import get_latest_checkpoint, save_checkpoint, resume
from datasets import build_train_loader
from posetimation.zoo import build_model
from posetimation.optimizer import build_lr_scheduler, build_optimizer
from posetimation.loss import build_loss
from engine.core import build_core_function
from engine.defaults import TRAIN_PHASE
from engine.apis import set_random_seed


class DefaultTrainer(BaseExecutor):
    def exec(self):
        self.train()

    def __init__(self, cfg, output_folders: dict, *args, **kwargs):
        super().__init__(cfg, output_folders, TRAIN_PHASE, **kwargs)
        logger = logging.getLogger(__name__)
        cfg = self.cfg

        logger.info("Set the random seed to {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

        self.dataloader = build_train_loader(cfg)
        self.model = build_model(cfg, phase='train')
        self.optimizer = build_optimizer(cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer)
        self.loss_criterion = build_loss(cfg)

        self.begin_epoch = 0
        self.end_epoch = cfg.TRAIN.END_EPOCH
        self.save_model_per_epoch = cfg.TRAIN.SAVE_MODEL_PER_EPOCH

        self.model = self.model.cuda()
        self.GPUS = cfg.GPUS
        self.core_function = build_core_function(cfg, criterion=self.loss_criterion, **kwargs)

        self.tb_writer_dict = {"writer": SummaryWriter(self.tb_save_folder),
                               "global_steps": 0}

    def train(self):
        logger = logging.getLogger(__name__)
        self.model_resume()
        if len(self.GPUS) > 1:
            self.model = torch.nn.DataParallel(self.model)
        for epoch in range(self.begin_epoch, self.end_epoch):
            # train
            logger.info('=> Start train epoch {}'.format(epoch))
            self.core_function.train(model=self.model, epoch=epoch, optimizer=self.optimizer, dataloader=self.dataloader,
                                     tb_writer_dict=self.tb_writer_dict)
            self.lr_scheduler.step(epoch)

            # save model
            if epoch % self.save_model_per_epoch == 0:
                model_save_path = self.save_model(epoch)
                logger.info('=> Saved epoch {} model state to {}'.format(epoch, model_save_path))

            # record learning_rate
            writer = self.tb_writer_dict["writer"]
            writer.add_scalar('learning_rate', self.lr_scheduler.get_lr(), epoch)

    def save_model(self, epoch):
        model_save_path = save_checkpoint(epoch, self.checkpoints_save_folder, self.model, self.optimizer,
                                          global_steps=self.tb_writer_dict["global_steps"])
        return model_save_path

    def model_resume(self):
        logger = logging.getLogger(__name__)
        checkpoint_file = get_latest_checkpoint(self.checkpoints_save_folder)
        if checkpoint_file is not None:
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            self.model, self.optimizer, self.begin_epoch, ext_dict = resume(self.model, self.optimizer, checkpoint_file, gpus=self.GPUS)
            self.tb_writer_dict["global_steps"] = ext_dict["tensorboard_global_steps"]

        else:
            logger.warning("=> no checkpoint file available to resume")

    def __del__(self):
        super(DefaultTrainer, self).__del__()
