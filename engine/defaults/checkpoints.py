#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import os.path as osp
from utils.utils_folder import list_immediate_childfile_paths


def get_latest_checkpoint(checkpoint_save_folder):
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None

    latest_checkpoint = checkpoint_saves_paths[0]
    # we define the format of checkpoint like "epoch_0_state.pth"
    latest_index = int(osp.basename(latest_checkpoint).split("_")[1])
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoint_save_file_name = osp.basename(checkpoint_save_path)
        now_index = int(checkpoint_save_file_name.split("_")[1])
        if now_index > latest_index:
            latest_checkpoint = checkpoint_save_path
            latest_index = now_index
    return latest_checkpoint


def get_all_checkpoints(checkpoint_save_folder):
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None
    checkpoints_list = []
    # we define the format of checkpoint like "epoch_0_state.pth"
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoints_list.append(checkpoint_save_path)
    return checkpoints_list


def save_checkpoint(epoch, save_folder, model, optimizer, **kwargs):
    model_save_path = osp.join(save_folder, 'epoch_{}_state.pth'.format(epoch))
    checkpoint_dict = dict()
    checkpoint_dict['begin_epoch'] = epoch

    # Because nn.DataParallel
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/5
    model_state_dict = model.state_dict()
    if list(model_state_dict.keys())[0].startswith('module.'):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}

    checkpoint_dict['state_dict'] = model_state_dict
    checkpoint_dict['optimizer'] = optimizer.state_dict()
    checkpoint_dict['tensorboard_global_steps'] = kwargs.get("global_steps", 0)
    torch.save(checkpoint_dict, model_save_path)

    return model_save_path


def resume(model, optimizer, checkpoint_file, **kwargs):
    ext_dict = {}
    checkpoint = torch.load(checkpoint_file)
    begin_epoch = checkpoint['begin_epoch'] + 1
    gpus = kwargs.get("gpus", [])
    if len(gpus) <= 1:
        state_dict = {k.replace('module.', '') if k.index('module') == 0 else k: v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    ext_dict["tensorboard_global_steps"] = checkpoint.get("tensorboard_global_steps", 0)

    return model, optimizer, begin_epoch, ext_dict
