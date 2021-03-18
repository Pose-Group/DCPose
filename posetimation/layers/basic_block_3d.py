#!/usr/bin/python
# -*- coding:utf8 -*-

import torch.nn as nn
from .basic_model import BasicBlock, BN_MOMENTUM


class Difference_aggreation_with_time(nn.Module):
    expansion = 1

    def __init__(self, input_channel, out_channel, kh, kw, dd, dg, num_blocks):
        super(Difference_aggreation_with_time, self).__init__()
        num_blocks = num_blocks
        block = BasicBlock
        head_conv_input_channel = input_channel
        body_conv_input_channel = out_channel
        body_conv_output_channel = out_channel
        stride = 1

        ######
        downsample = nn.Sequential(
            nn.Conv2d(
                head_conv_input_channel,
                body_conv_input_channel,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(
                body_conv_input_channel,
                momentum=BN_MOMENTUM
            ),
        )

        # ##########3
        layers = []
        layers.append(
            block(
                head_conv_input_channel,
                body_conv_input_channel,
                stride,
                downsample
            )
        )

        for i in range(1, num_blocks):
            layers.append(
                block(
                    body_conv_input_channel,
                    body_conv_output_channel
                )
            )
        self.modlist = nn.ModuleList(layers)

    def forward(self, x):
        for m in self.modlist:
            x = m(x)
        return x
