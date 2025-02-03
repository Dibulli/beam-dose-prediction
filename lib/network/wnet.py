import torch
import torch.nn as nn


class Wnet(nn.Module):
    def __init__(self, input_channels, mid_channels, out_channels, init_weights=True):
        super(Wnet, self).__init__()
        self.u1 = UNet(input_channels, mid_channels, sigmoid_out=True, init_weights=True)
        self.u2 = UNet(mid_channels + input_channels, out_channels, sigmoid_out=False, init_weights=True)

    def forward(self, x):
        x_encoded = self.u1(x)
        x_merge = torch.cat((x_encoded, x), 1)
        x_decoded = self.u2(x_merge)
        return x_encoded, x_decoded



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, sigmoid_out=True, init_weights=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)

        self.gating_feature0 = IniGatingFeature(256)
        self.gating_feature1 = GatingFeature(256)
        self.gating_feature2 = GatingFeature(128)
        self.gating_feature3 = GatingFeature(64)
        self.gating_feature4 = GatingFeature(32)

        self.gating_feature0_up = Up(256, 256)
        self.gating_feature1_up = Up(256, 128)
        self.gating_feature2_up = Up(128, 64)
        self.gating_feature3_up = Up(64, 32)

        self.feature0_up = Up(256, 256)
        self.feature1_up = Up(128, 128)
        self.feature2_up = Up(64, 64)
        self.feature3_up = Up(32, 32)

        self.feature0_conv = DoubleConv(256, 256)
        self.feature1_conv = DoubleConv(256, 128)
        self.feature2_conv = DoubleConv(128, 64)
        self.feature3_conv = DoubleConv(64, 32)
        self.feature4_conv = DoubleConv(32, 32)
        
        self.sigmoid_out = sigmoid_out

        self.outc = OutConv(32, n_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        gating_feature_0 = self.gating_feature0(x5)
        gating_feature_0_up = self.gating_feature0_up(gating_feature_0)

        gating_feature_1 = self.gating_feature1(x4, gating_feature_0_up)
        gating_feature_1_up = self.gating_feature1_up(gating_feature_1)

        gating_feature_2 = self.gating_feature2(x3, gating_feature_1_up)
        gating_feature_2_up = self.gating_feature2_up(gating_feature_2)

        gating_feature_3 = self.gating_feature3(x2, gating_feature_2_up)
        gating_feature_3_up = self.gating_feature3_up(gating_feature_3)

        gating_feature_4 = self.gating_feature4(x1, gating_feature_3_up)

        gating_signal_0 = torch.sigmoid(gating_feature_0)
        gating_signal_1 = torch.sigmoid(gating_feature_1)
        gating_signal_2 = torch.sigmoid(gating_feature_2)
        gating_signal_3 = torch.sigmoid(gating_feature_3)
        gating_signal_4 = torch.sigmoid(gating_feature_4)

        x = self.feature0_conv(gating_signal_0 * x5)
        x = self.feature0_up(x)
        x = self.feature1_conv(gating_signal_1 * torch.add(x, x4))
        x = self.feature1_up(x)
        x = self.feature2_conv(gating_signal_2 * torch.add(x, x3))
        x = self.feature2_up(x)
        x = self.feature3_conv(gating_signal_3 * torch.add(x, x2))
        x = self.feature3_up(x)
        x = self.feature4_conv(gating_signal_4 * torch.add(x, x1))
        x = self.outc(x)
        if self.sigmoid_out:
          return torch.sigmoid(x)
        else:
          return nn.ReLU()(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class IniGatingFeature(nn.Module):
    def __init__(self, in_ch):
        super(IniGatingFeature, self).__init__()
        self.conv = DoubleConv(in_ch, in_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class GatingFeature(nn.Module):
    def __init__(self, in_ch):
        super(GatingFeature, self).__init__()
        self.conv1 = DoubleConv(in_ch, in_ch)
        self.conv2 = DoubleConv(in_ch, in_ch)

    def forward(self, x0, x1):
        x = self.conv1(x0)
        x = torch.add(x, x1)
        x = self.conv2(x)
        return x