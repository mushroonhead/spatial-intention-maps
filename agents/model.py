# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import SinusoidalPosEmb
from typing import Union


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        #input_dim = 16 + 9216 + 5*96*96 #state_dim + action_dim + t_dim
        input_dim = state_dim + action_dim + t_dim

        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 256),
                                       nn.ReLU())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state: torch.Tensor):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state.flatten(start_dim=-3)], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class Interpolate(torch.nn.Module):
    """
    Wrapper for interpolate function
    """
    def __init__(self, interp_args: dict,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.interp_args = interp_args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, **self.interp_args)


class LightUnet(torch.nn.Module):
    """
    Lightweight resnet for small datasets
    """
    def __init__(self, inp_channel: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.down1 = torch.nn.Sequential(
            torch.nn.Conv2d(inp_channel, inp_channel, kernel_size=2, stride=2, groups=inp_channel),
            torch.nn.BatchNorm2d(inp_channel),
            torch.nn.Mish(),
            torch.nn.Conv2d(inp_channel, 32, kernel_size=1, stride=1))
        self.down2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=3, groups=32),
            torch.nn.BatchNorm2d(32),
            torch.nn.Mish(),
            torch.nn.Conv2d(32, 64, kernel_size=1, stride=1))
        self.up2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.Mish(),
            Interpolate({'scale_factor':3, 'mode':'bilinear', 'align_corners':True}))
        self.up1 = torch.nn.Sequential(
            torch.nn.Conv2d(128,  256, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.Mish(),
            Interpolate({'scale_factor':2, 'mode':'bilinear', 'align_corners':True}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x) # (B,32,H,W)
        x2 = self.down2(x1) # (B,64,H,W)
        x1 = self.up2(x2) #(B,128,H,W)
        x = self.up1(x1) # (B,256,H,W)

        return x

class LightConditionalNetwork(torch.nn.Module):
    def __init__(self, inp_channel: int, cond_channel: int,
                 t_dim : int=16,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cond_channel = cond_channel
        self.t_dim = t_dim
        self.inp_encoder = LightUnet(inp_channel)
        self.time_encoder = SinusoidalPosEmb(t_dim)
        self.shared_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(256+cond_channel+t_dim, 128, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.Mish(),
            torch.nn.Conv2d(128, 64, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.Mish(),
            torch.nn.Conv2d(64, 32, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.Mish(),
            torch.nn.Conv2d(32, inp_channel, kernel_size=1, stride=1))

    def forward(self, sample: torch.Tensor,
                timestep: Union[int, torch.IntTensor],
                encoder_hidden_states: torch.Tensor) -> torch.Tensor:

        sample_features = self.inp_encoder(sample) # (B,256,H,W)\

        if isinstance(timestep, int):
            timestep = torch.tensor((timestep,), device=sample.device) #(1,)
        elif timestep.ndim < 1:
            timestep = timestep.unsqueeze(-1) #(1,)
        timestep = timestep.to(device=sample.device)
        time_encoding = self.time_encoder(timestep).view(-1,self.t_dim).expand(sample.shape[0], -1) # (B,t_dim)
        time_encoding = time_encoding[...,None,None].expand(-1,-1,*sample_features.shape[-2:]) # (B,t_dim,H,W)

        encoder_hidden_states = encoder_hidden_states.view(encoder_hidden_states.shape[0], -1) # (B,cond_c)
        encoder_hidden_states = encoder_hidden_states[...,None,None].expand(-1,-1,*sample_features.shape[-2:]) # (B,cond_c,H,W)
        output = self.shared_mlp(
            torch.cat((sample_features, encoder_hidden_states, time_encoding), dim=-3)) # (B,total,H,W)

        return output # (B,inp_c,H,W)

if __name__ == '__main__':
    x = torch.randn(32, 1, 96, 96)
    t = torch.randint(0, 100, (32,))
    cond = torch.randn(32,1,256)

    x_encoded = LightUnet(1)(x)
    print(x_encoded.shape)

    y = LightConditionalNetwork(1, 256)(sample=x, timestep=t,
                                        encoder_hidden_states=cond)
    print(y.shape)