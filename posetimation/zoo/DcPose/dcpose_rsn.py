#!/usr/bin/python
# -*- coding:utf8 -*-


import os
import torch
import torch.nn as nn
import logging
from collections import OrderedDict

from ..base import BaseModel

from thirdparty.deform_conv import DeformConv, ModulatedDeformConv
from posetimation.layers import BasicBlock, ChainOfBasicBlocks, DeformableCONV, PAM_Module, CAM_Module
from posetimation.layers import RSB_BLOCK, CHAIN_RSB_BLOCKS
from ..backbones.hrnet import HRNet
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DcPose_RSN(BaseModel):

    def __init__(self, cfg, phase, **kwargs):
        super(DcPose_RSN, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.inplanes = 64
        self.use_warping_train = cfg['MODEL']['USE_WARPING_TRAIN']
        self.use_warping_test = cfg['MODEL']['USE_WARPING_TEST']
        self.freeze_weights = cfg['MODEL']['FREEZE_WEIGHTS']
        self.use_gt_input_train = cfg['MODEL']['USE_GT_INPUT_TRAIN']
        self.use_gt_input_test = cfg['MODEL']['USE_GT_INPUT_TEST']
        self.warping_reverse = cfg['MODEL']['WARPING_REVERSE']
        self.cycle_consistency_finetune = cfg['MODEL']['CYCLE_CONSISTENCY_FINETUNE']

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        self.is_train = True if phase == TRAIN_PHASE else False
        # define rough_pose_estimation
        self.use_prf = cfg.MODEL.USE_PRF
        self.use_ptm = cfg.MODEL.USE_PTM
        self.use_pcn = cfg.MODEL.USE_PCN

        self.freeze_hrnet_weights = cfg.MODEL.FREEZE_HRNET_WEIGHTS
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.use_rectifier = cfg.MODEL.USE_RECTIFIER
        self.use_margin = cfg.MODEL.USE_MARGIN
        self.use_group = cfg.MODEL.USE_GROUP

        self.deformable_conv_dilations = cfg.MODEL.DEFORMABLE_CONV.DILATION
        self.deformable_aggregation_type = cfg.MODEL.DEFORMABLE_CONV.AGGREGATION_TYPE
        ####
        self.rough_pose_estimation_net = HRNet(cfg, phase)

        self.pretrained = cfg.MODEL.PRETRAINED

        k = 3

        prf_inner_ch = cfg.MODEL.PRF_INNER_CH
        prf_basicblock_num = cfg.MODEL.PRF_BASICBLOCK_NUM

        ptm_inner_ch = cfg.MODEL.PTM_INNER_CH
        ptm_basicblock_num = cfg.MODEL.PTM_BASICBLOCK_NUM

        prf_ptm_combine_inner_ch = cfg.MODEL.PRF_PTM_COMBINE_INNER_CH
        prf_ptm_combine_basicblock_num = cfg.MODEL.PRF_PTM_COMBINE_BASICBLOCK_NUM
        hyper_parameters = OrderedDict({
            "k": k,
            "prf_basicblock_num": prf_basicblock_num,
            "prf_inner_ch": prf_inner_ch,
            "ptm_basicblock_num": ptm_basicblock_num,
            "ptm_inner_ch": ptm_inner_ch,
            "prf_ptm_combine_basicblock_num": prf_ptm_combine_basicblock_num,
            "prf_ptm_combine_inner_ch": prf_ptm_combine_inner_ch,
        }
        )
        self.logger.info("###### MODEL {} Hyper Parameters ##########".format(self.__class__.__name__))
        self.logger.info(hyper_parameters)

        assert self.use_prf and self.use_ptm and self.use_pcn and self.use_margin and self.use_margin and self.use_group

        ####### PRF #######
        diff_temporal_fuse_input_channels = self.num_joints * 4
        self.diff_temporal_fuse = CHAIN_RSB_BLOCKS(diff_temporal_fuse_input_channels, prf_inner_ch, prf_basicblock_num,
                                                   )

        # self.diff_temporal_fuse = ChainOfBasicBlocks(diff_temporal_fuse_input_channels, prf_inner_ch, 1, 1, 2,
        #                                              prf_basicblock_num, groups=self.num_joints)

        ####### PTM #######
        if ptm_basicblock_num > 0:

            self.support_temporal_fuse = CHAIN_RSB_BLOCKS(self.num_joints * 3, ptm_inner_ch, ptm_basicblock_num,
                                                          )

            # self.support_temporal_fuse = ChainOfBasicBlocks(self.num_joints * 3, ptm_inner_ch, 1, 1, 2,
            #                                                 ptm_basicblock_num, groups=self.num_joints)
        else:
            self.support_temporal_fuse = nn.Conv2d(self.num_joints * 3, ptm_inner_ch, kernel_size=3, padding=1,
                                                   groups=self.num_joints)

        prf_ptm_combine_ch = prf_inner_ch + ptm_inner_ch

        self.offset_mask_combine_conv = CHAIN_RSB_BLOCKS(prf_ptm_combine_ch, prf_ptm_combine_inner_ch, prf_ptm_combine_basicblock_num)
        # self.offset_mask_combine_conv = ChainOfBasicBlocks(prf_ptm_combine_ch, prf_ptm_combine_inner_ch, 1, 1, 2,
        #                                                    prf_ptm_combine_basicblock_num)

        ####### PCN #######
        self.offsets_list, self.masks_list, self.modulated_deform_conv_list = [], [], []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            # offsets
            offset_layers, mask_layers = [], []
            offset_layers.append(self._offset_conv(prf_ptm_combine_inner_ch, k, k, dilation, self.num_joints).cuda())
            mask_layers.append(self._mask_conv(prf_ptm_combine_inner_ch, k, k, dilation, self.num_joints).cuda())
            self.offsets_list.append(nn.Sequential(*offset_layers))
            self.masks_list.append(nn.Sequential(*mask_layers))
            self.modulated_deform_conv_list.append(DeformableCONV(self.num_joints, k, dilation))

        self.offsets_list = nn.ModuleList(self.offsets_list)
        self.masks_list = nn.ModuleList(self.masks_list)
        self.modulated_deform_conv_list = nn.ModuleList(self.modulated_deform_conv_list)

    def _offset_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 2 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd), padding=(1 * dd, 1 * dd), bias=False)
        return conv

    def _mask_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 1 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd), padding=(1 * dd, 1 * dd), bias=False)
        return conv

    # def forward(self, x, margin, debug=False, vis=False):
    def forward(self, x, **kwargs):
        num_color_channels = 3
        assert "margin" in kwargs
        margin = kwargs["margin"]
        if not x.is_cuda or not margin.is_cuda:
            x.cuda()
            margin.cuda()

        if not self.use_rectifier:
            target_image = x[:, 0:num_color_channels, :, :]
            rough_x = self.rough_pose_estimation_net(target_image)
            return rough_x

        # current / previous / next
        rough_heatmaps = self.rough_pose_estimation_net(torch.cat(x.split(num_color_channels, dim=1), 0))
        true_batch_size = int(rough_heatmaps.shape[0] / 3)

        # rough heatmaps in sequence frames
        current_rough_heatmaps, previous_rough_heatmaps, next_rough_heatmaps = rough_heatmaps.split(true_batch_size, dim=0)

        # Difference A and Difference B
        diff_A = current_rough_heatmaps - previous_rough_heatmaps
        diff_B = current_rough_heatmaps - next_rough_heatmaps

        # default use_margin,use_group
        interval = torch.sum(margin, dim=1, keepdim=True)  # interval = n-p
        margin = torch.div(margin.float(), interval.float())  # margin -> (c-p)/(n-p), (n-c)/(n-p)
        prev_weight, next_weight = margin[:, 1], margin[:, 0]  # previous frame weight - (n-c)/(n-p) , next frame weight - (c-p)/(n-p)
        diff_shape = diff_A.shape
        prev_weight = prev_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        next_weight = next_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        extend_shape = [1, 1]  # batch, channel
        extend_shape.extend(list(diff_shape[2:]))
        prev_weight, next_weight = prev_weight.repeat(extend_shape), next_weight.repeat(extend_shape)

        diff_A_list, diff_B_list = diff_A.split(1, dim=1), diff_B.split(1, dim=1)
        temp_diff_fuse_list = []
        for joint_index in range(self.num_joints):
            temp_diff_fuse_list.append(diff_A_list[joint_index])
            temp_diff_fuse_list.append(diff_A_list[joint_index] * prev_weight)
            temp_diff_fuse_list.append(diff_B_list[joint_index])
            temp_diff_fuse_list.append(diff_B_list[joint_index] * next_weight)

        dif_heatmaps = torch.cat(temp_diff_fuse_list, dim=1)
        dif_heatmaps = self.diff_temporal_fuse(dif_heatmaps)

        current_rough_heatmaps_list = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmaps_list = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmaps_list = next_rough_heatmaps.split(1, dim=1)
        temp_support_fuse_list = []
        for joint_index in range(self.num_joints):
            temp_support_fuse_list.append(current_rough_heatmaps_list[joint_index])
            temp_support_fuse_list.append(previous_rough_heatmaps_list[joint_index] * prev_weight)
            temp_support_fuse_list.append(next_rough_heatmaps_list[joint_index] * next_weight)

        support_heatmaps = torch.cat(temp_support_fuse_list, dim=1)
        support_heatmaps = self.support_temporal_fuse(support_heatmaps).cuda()

        prf_ptm_combine_featuremaps = self.offset_mask_combine_conv(torch.cat([dif_heatmaps, support_heatmaps], dim=1))

        warped_heatmaps_list = []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            offsets = self.offsets_list[d_index](prf_ptm_combine_featuremaps)
            masks = self.masks_list[d_index](prf_ptm_combine_featuremaps)

            warped_heatmaps = self.modulated_deform_conv_list[d_index](support_heatmaps, offsets, masks)

            warped_heatmaps_list.append(warped_heatmaps)

        if self.deformable_aggregation_type == "weighted_sum":

            warper_weight = 1 / len(self.deformable_conv_dilations)
            output_heatmaps = warper_weight * warped_heatmaps_list[0]
            for warper_heatmaps in warped_heatmaps_list[1:]:
                output_heatmaps += warper_weight * warper_heatmaps

        else:
            output_heatmaps = self.deformable_aggregation_conv(torch.cat(warped_heatmaps_list, dim=1))
            # elif self.deformable_aggregation_type == "conv":

        if not self.freeze_hrnet_weights:
            return current_rough_heatmaps, output_heatmaps
        else:
            return output_heatmaps

    def init_weights(self):
        logger = logging.getLogger(__name__)
        ## init_weights
        rough_pose_estimation_name_set = set()
        for module_name, module in self.named_modules():
            # rough_pose_estimation_net 单独判断一下
            if module_name.split('.')[0] == "rough_pose_estimation_net":
                rough_pose_estimation_name_set.add(module_name)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, DeformConv):
                filler = torch.zeros([module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
                                     dtype=torch.float32, device=module.weight.device)
                for k in range(module.weight.size(0)):
                    filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
                module.weight = torch.nn.Parameter(filler)
                # module.weight.requires_grad = True
            elif isinstance(module, ModulatedDeformConv):
                filler = torch.zeros([module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
                                     dtype=torch.float32, device=module.weight.device)
                for k in range(module.weight.size(0)):
                    filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
                module.weight = torch.nn.Parameter(filler)
                # module.weight.requires_grad = True
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        if os.path.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('=> loading pretrained model {}'.format(self.pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    layer_name = name.split('.')[0]
                    if layer_name in rough_pose_estimation_name_set:
                        need_init_state_dict[name] = m
                    else:
                        # 为了适应原本hrnet得预训练网络
                        new_layer_name = "rough_pose_estimation_net.{}".format(layer_name)
                        if new_layer_name in rough_pose_estimation_name_set:
                            parameter_name = "rough_pose_estimation_net.{}".format(name)
                            need_init_state_dict[parameter_name] = m
            # TODO pretrained from posewarper not test
            self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(self.pretrained))

        # rough_pose_estimation
        if self.freeze_hrnet_weights:
            self.rough_pose_estimation_net.freeze_weight()

    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):
        prf_inner_ch = cfg.MODEL.PRF_INNER_CH
        prf_basicblock_num = cfg.MODEL.PRF_BASICBLOCK_NUM
        ptm_inner_ch = cfg.MODEL.PTM_INNER_CH
        ptm_basicblock_num = cfg.MODEL.PTM_BASICBLOCK_NUM
        prf_ptm_combine_inner_ch = cfg.MODEL.PRF_PTM_COMBINE_INNER_CH
        prf_ptm_combine_basicblock_num = cfg.MODEL.PRF_PTM_COMBINE_BASICBLOCK_NUM
        if "DILATION" in cfg.MODEL.DEFORMABLE_CONV:
            dilation = cfg.MODEL.DEFORMABLE_CONV.DILATION
            dilation_str = ",".join(map(str, dilation))
        else:
            dilation_str = ""
        hyper_parameters_setting = "chPRF_{}_nPRF_{}_chPTM_{}_nPTM_{}_chComb_{}_nComb_{}_D_{}".format(
            prf_inner_ch, prf_basicblock_num, ptm_inner_ch, ptm_basicblock_num, prf_ptm_combine_inner_ch, prf_ptm_combine_basicblock_num,
            dilation_str)


        return hyper_parameters_setting

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = DcPose_RSN(cfg, phase, **kwargs)
        return model
