import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet


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

if __name__ == '__main__':
    C = 5
    x = torch.randn(32,C,96,96)
    model = LightWeightBottleneck(C)
    print(model(x).shape)