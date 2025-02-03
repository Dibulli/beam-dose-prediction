import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_weights=True):
        super(UNet, self).__init__()
        self.down1 = DonwSampleBlock2(n_channels, 64)
        self.down2 = DonwSampleBlock2(64, 128)
        self.down3 = DonwSampleBlock3(128, 256)
        self.down4 = DonwSampleBlock3(256, 512)
        self.down5 = DonwSampleBlock3(512, 1024)

        self.up4 = UpSampleBlock(1024, 512)
        self.up3 = UpSampleBlock(512, 256)
        self.up2 = UpSampleBlock(256, 128)
        self.up1 = UpSampleBlock(128, 64)
        self.out = OutBlock(n_channels, 64, n_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.down1(x)      # 64, 256, 256
        x2 = self.down2(x1)     # 128, 128, 128
        x3 = self.down3(x2)     # 256, 64, 64
        x4 = self.down4(x3)     # 512, 32, 32
        x5 = self.down5(x4)     # 1024, 16, 16
        x4 = self.up4(x5, x4)
        x3 = self.up3(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up1(x2, x1)
        x = self.out(x1, x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DonwSampleBlock2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DonwSampleBlock2, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class DonwSampleBlock3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DonwSampleBlock3, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.add(x1, x2)
        x = self.conv(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_ch, intermediate_ch, n_classes):
        super(OutBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + intermediate_ch, intermediate_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_ch, intermediate_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_ch, n_classes, 1, padding=0),
        )

    def forward(self, x1, x2):
        x1 = nn.UpsamplingBilinear2d(scale_factor=(2, 2))(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels, n_classes, init_weights=True):
        super(Encoder, self).__init__()
        self.down1 = DonwSampleBlock2(n_channels, 64)
        self.down2 = DonwSampleBlock2(64, 128)
        self.down3 = DonwSampleBlock3(128, 256)
        self.down4 = DonwSampleBlock3(256, 512)
        self.down5 = DonwSampleBlock3(512, 1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, n_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.down1(x)      # 64, 64, 64
        x2 = self.down2(x1)     # 128, 32, 32
        x3 = self.down3(x2)     # 256, 16, 16
        x4 = self.down4(x3)     # 512, 8, 8
        x5 = self.down5(x4)     # 1024, 4, 4
        x = self.avgpool(x5)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


