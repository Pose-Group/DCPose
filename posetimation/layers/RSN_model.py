#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

__all__ = ["RSB_BLOCK", "CHAIN_RSB_BLOCKS", "RSN_WEIGHT_VECTOR", "RSN_ATTENTION"]


class CHAIN_RSB_BLOCKS(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks, groups=1):
        super(CHAIN_RSB_BLOCKS, self).__init__()
        layers = []
        stride = 1
        downSample = conv_bn_relu(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, has_relu=False, groups=groups)
        layers.append(RSB_BLOCK(in_planes, out_planes, stride, downsample=downSample))
        for i in range(1, num_blocks):
            layers.append(RSB_BLOCK(out_planes, out_planes, stride, downsample=None))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RSB_BLOCK(nn.Module):
    """
    from https://github.com/caiyuanhao1998/RSN/blob/master/exps/4XRSN18.coco/network.py
        class Bottleneck(nn.Module)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, groups=1, downsample=None, efficient=False):
        super(RSB_BLOCK, self).__init__()
        self.branch_ch = in_planes * 26 // 64
        self.conv_bn_relu1 = conv_bn_relu(in_planes, 4 * self.branch_ch, kernel_size=1,
                                          stride=stride, padding=0, groups=groups,
                                          has_bn=True, has_relu=True, efficient=efficient)

        self.conv_bn_relu2_1_1 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_2_1 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_2_2 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_3_1 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_3_2 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_3_3 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_4_1 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_4_2 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_4_3 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_4_4 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)

        self.conv_bn_relu3 = conv_bn_relu(4 * self.branch_ch, planes * self.expansion,
                                          kernel_size=1, stride=1, padding=0, groups=groups,
                                          has_bn=True, has_relu=False, efficient=efficient)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        spx = torch.split(out, self.branch_ch, 1)
        out_1_1 = self.conv_bn_relu2_1_1(spx[0])

        out_2_1 = self.conv_bn_relu2_2_1(spx[1] + out_1_1)
        out_2_2 = self.conv_bn_relu2_2_2(out_2_1)

        out_3_1 = self.conv_bn_relu2_3_1(spx[2] + out_2_1)
        out_3_2 = self.conv_bn_relu2_3_2(out_3_1 + out_2_2)
        out_3_3 = self.conv_bn_relu2_3_3(out_3_2)

        out_4_1 = self.conv_bn_relu2_4_1(spx[3] + out_3_1)
        out_4_2 = self.conv_bn_relu2_4_2(out_4_1 + out_3_2)
        out_4_3 = self.conv_bn_relu2_4_3(out_4_2 + out_3_3)
        out_4_4 = self.conv_bn_relu2_4_4(out_4_3)

        out = torch.cat((out_1_1, out_2_2, out_3_3, out_4_4), 1)
        out = self.conv_bn_relu3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out


class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,
                 has_bn=True, has_relu=True, efficient=False, groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient

        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x

            return func

        func = _func_factory(self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x


class RSN_WEIGHT_VECTOR(nn.Module):

    def __init__(self, input_chn_num, output_chl_num):
        super(RSN_WEIGHT_VECTOR, self).__init__()
        self.input_chm_num = input_chn_num
        self.output_chl_num = output_chl_num

        self.conv_bn_relu_1 = conv_bn_relu(self.input_chm_num, self.output_chl_num, kernel_size=3,
                                           stride=1, padding=1, has_bn=True, has_relu=True)

        self.conv_bn_relu_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                           stride=1, padding=0, has_bn=True, has_relu=True)
        self.conv_bn_relu_3 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                           stride=1, padding=0, has_bn=True, has_relu=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_bn_relu_1(x)
        out_0 = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out_1 = self.conv_bn_relu_2(out_0)
        out_2 = self.conv_bn_relu_3(out_1+out_0)
        out_3 = self.sigmoid(out_2)

        return out_3


class RSN_ATTENTION(nn.Module):

    def __init__(self, output_chl_num, efficient=False):
        super(RSN_ATTENTION, self).__init__()
        self.output_chl_num = output_chl_num
        self.conv_bn_relu_prm_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=3,
                                               stride=1, padding=1, has_bn=True, has_relu=True,
                                               efficient=efficient)
        self.conv_bn_relu_prm_2_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                                 stride=1, padding=0, has_bn=True, has_relu=True,
                                                 efficient=efficient)
        self.conv_bn_relu_prm_2_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                                 stride=1, padding=0, has_bn=True, has_relu=True,
                                                 efficient=efficient)
        self.sigmoid2 = nn.Sigmoid()
        self.conv_bn_relu_prm_3_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                                 stride=1, padding=0, has_bn=True, has_relu=True,
                                                 efficient=efficient)

        self.conv_bn_relu_prm_3_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=9,
                                                 stride=1, padding=4, has_bn=True, has_relu=True,
                                                 efficient=efficient, groups=self.output_chl_num)
        # 特征图大小都不变
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_bn_relu_prm_1(x)
        out_1 = out
        out_2 = torch.nn.functional.adaptive_avg_pool2d(out_1, (1, 1))  # Global Pooling  (N,C,1,1)
        out_2 = self.conv_bn_relu_prm_2_1(out_2)
        out_2 = self.conv_bn_relu_prm_2_2(out_2)
        out_2 = self.sigmoid2(out_2)
        out_3 = self.conv_bn_relu_prm_3_1(out_1)
        out_3 = self.conv_bn_relu_prm_3_2(out_3)
        out_3 = self.sigmoid3(out_3)  # like
        out = out_1.mul(1 + out_2.mul(out_3))
        return out
