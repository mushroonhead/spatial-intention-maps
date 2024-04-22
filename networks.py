import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

import resnet
from agents.helpers import SinusoidalPosEmb
from agents.model import MLP

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


class FCN(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super().__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, num_output_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.resnet18.features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv3(x)

class LightEncoderFCN(torch.nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=256):
        super().__init__()
        self.resnet = resnet.resnet_N(layers=[2,2,2],num_input_channels=num_input_channels,
                                      features_only=True)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_output_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        return self.conv2(x)

class LightWeightBottleneck(torch.nn.Module):
    """
    Shrinks CxHxW to 1xF where F << C*H*W
    - Consist of layer wise and channel wise indep convolutions
    """
    def __init__(self, inp_channel: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(inp_channel, inp_channel, kernel_size=2, stride=2, groups=inp_channel),
            torch.nn.BatchNorm2d(inp_channel),
            torch.nn.Mish(),
            torch.nn.Conv2d(inp_channel, inp_channel>>1, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(inp_channel>>1),
            torch.nn.Mish(),
            torch.nn.Conv2d(inp_channel>>1, inp_channel>>1, kernel_size=3, stride=3, groups=inp_channel>>1),
            torch.nn.BatchNorm2d(inp_channel>>1),
            torch.nn.Mish(),
            torch.nn.Conv2d(inp_channel>>1, inp_channel>>2, kernel_size=1, stride=1),
            torch.nn.Flatten(-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
                 repeat_enc = True,
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
        self.expand_enc = self._expand_enc_repeat_pixels if repeat_enc \
            else self._expand_enc_align_pixels

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

        encoder_hidden_states = self.expand_enc(encoder_hidden_states, *sample_features.shape[-2:]) # (B,cond_c,H,W)
        output = self.shared_mlp(
            torch.cat((sample_features, encoder_hidden_states, time_encoding), dim=-3)) # (B,total,H,W)

        return output # (B,inp_c,H,W)

    def _expand_enc_repeat_pixels(self, encoder_hidden_states: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        - Inputs:
            - encoder_hidden_states: (B,cond_c) tensor
        - Returns:
            - encoder_hidden_states: (B,cond_c,h,w) tensor
        """
        encoder_hidden_states = encoder_hidden_states.view(encoder_hidden_states.shape[0], -1) # (B,cond_c)
        encoder_hidden_states = encoder_hidden_states[...,None,None].expand(-1,-1,h,w) # (B,cond_c,H,W)\
        return encoder_hidden_states

    def _expand_enc_align_pixels(self, encoder_hidden_states: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        - Inputs:
            - encoder_hidden_states: (B,cond_c,h,w) tensor
        - Returns:
            - encoder_hidden_states: (B,cond_c,h,w) tensor
        """
        return encoder_hidden_states

class CondMLP(torch.nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 t_dim=16,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.a_mlp = torch.nn.Sequential(
            torch.nn.Linear(action_dim, 32),
            torch.nn.Mish(),
            torch.nn.Linear(32, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, 128),
        )
        input_dim = state_dim + 128 + t_dim
        self.bot_layer = nn.Sequential(nn.Linear(input_dim, 1024),
                                       nn.Mish(),
                                       nn.Linear(1024, 128),
                                       nn.Mish(),
                                       nn.Linear(128, 64),
                                       nn.Mish(),
                                       nn.Linear(64, action_dim))

    def forward(self, sample: torch.Tensor, timestep: torch.IntTensor,
                encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        B = sample.shape[0]
        x_enc = self.a_mlp(sample)
        if timestep.ndim < 1:
            timestep = timestep.unsqueeze(-1) #(1,)
        t_enc = self.time_mlp(timestep.to(sample.device)).expand(B,-1)
        x = torch.cat([x_enc, t_enc, encoder_hidden_states.flatten(start_dim=-3)], dim=1)
        x = self.bot_layer(x)

        return x

################### Image net 2

class LightCondUnet2D(torch.nn.Module):
    def __init__(self,
                    encoder_channel: int,
                    inp_channel: int,
                    t_dim: int,
                    *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a_enc = resnet.resnet_N(layers=[2,2,2], features_only=True,
                                        num_input_channels=inp_channel)
        self.time_enc = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim))
        stacked_channels = 256 + encoder_channel + t_dim # 256 det by layers=[2,2,2]
        self.dec = nn.Sequential(
            torch.nn.Conv2d(stacked_channels, 256, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.Mish(),
            Interpolate({'scale_factor':2, 'mode':'bilinear', 'align_corners':True}),
            torch.nn.Conv2d(256, 64, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.Mish(),
            Interpolate({'scale_factor':2, 'mode':'bilinear', 'align_corners':True}),
            torch.nn.Conv2d(64, inp_channel, kernel_size=1, stride=1))

    def forward(self, sample: torch.Tensor, timestep: torch.IntTensor,
                encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        a_enc = self.a_enc(sample) # (B,a_enc,24,24)
        if timestep.ndim < 1:
            timestep = timestep.unsqueeze(-1) #(1,)
        t_enc = self.time_enc(timestep.to(sample.device))[...,None,None].expand(
            a_enc.shape[0],-1,*a_enc.shape[-2:]) # (B,t_dim,24,24)
        x = torch.cat((a_enc, encoder_hidden_states, t_enc), dim=-3)
        x = self.dec(x)

        return x

if __name__ == '__main__':



    # # light bottleneck
    # C = 5
    # x = torch.randn(32,C,96,96)
    # model = LightWeightBottleneck(C)(x)

    # x.shape

    ############################

    # # test LightUnet
    # C = 5
    # x = torch.randn(32,C,96,96)
    # model = LightWeightBottleneck(C)
    # print(model(x).shape)

    # # test LightConditionalNetwork

    # x = torch.randn(32, 1, 96, 96)
    # t = torch.randint(0, 100, (32,))
    # cond = torch.randn(32,1,256)

    # x_encoded = LightUnet(1)(x)
    # print(x_encoded.shape)

    # y = LightConditionalNetwork(1, 256)(sample=x, timestep=t,
    #                                     encoder_hidden_states=cond)
    # print(y.shape)


    ##########################
    # C = 5

    # x = torch.randn(32, C, 96, 96)

    # res = resnet.resnet18(num_input_channels=C)
    # y = res(x)
    # print(y.shape)

    # resf = resnet.resnet_N(layers=[2,2,2], features_only=True, num_input_channels=C)
    # y2 = resf(x)
    # print(y2.shape)

    # C = 5

    # x = torch.randn(32, C, 96, 96)

    # lfcn = LightEncoderFCN(C)
    # y = lfcn(x)
    # print(y.shape)

    ###################

    # state = torch.randn(32, 5, 96,96)
    # action = torch.randn(32,2)
    # t = torch.randint(0,100, (32,))

    # resf = resnet.resnet_N(layers=[2,], features_only=True, num_input_channels=5)
    # enc = resf(state)
    # cond_mlp = CondMLP(64*24*24, 2, 16)
    # pred_noise = cond_mlp(sample=action, timestep=t,
    #                       encoder_hidden_states=enc)

    # print(pred_noise.shape)


    ############# Image net 2

    state = torch.randn(32, 5, 96,96)
    action = torch.randn(32, 1, 96,96)
    t = torch.randint(0,100, (32,))

    resf = resnet.resnet_N(layers=[2,2,2], features_only=True, num_input_channels=5)
    enc = resf(state)
    cond_mlp = LightCondUnet2D(256,1,16)
    pred_noise = cond_mlp(sample=action, timestep=t,
                          encoder_hidden_states=enc)

    print(pred_noise.shape)


    pass