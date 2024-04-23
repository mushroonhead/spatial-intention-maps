# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet

from agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 fcn_input_channels,
                 fcn_output_channels,
                 one_channel_state_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.resnet18 = resnet.resnet18(num_input_channels=fcn_input_channels)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, fcn_output_channels, kernel_size=1, stride=1)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        #input_dim = 16 + 9216 + 5*96*96 #state_dim + action_dim + t_dim
        input_dim = fcn_output_channels*one_channel_state_dim + action_dim + t_dim
        
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    
    def state_through_resnet(self, state):
        state = self.resnet18.features(state)
        state = self.conv1(state)
        state = self.bn1(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        state = self.conv2(state)
        state = self.bn2(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        state = self.conv3(state)
        if state.shape[1] != 1 or state.shape[2] != 96 or state.shape[3] != 96:
            raise Exception("ERROR: state dimension is not batch_size * 1 * 96 * 96, error is in model.py")

        return state
    
    def forward(self, x, time, state: torch.Tensor):

        t = self.time_mlp(time)
        #print(x.shape, t.shape, state.flatten(start_dim=-3).shape)
        x = torch.cat([x, t, state.flatten(start_dim=-3)], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


