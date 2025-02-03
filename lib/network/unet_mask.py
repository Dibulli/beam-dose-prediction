import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_weights=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)

        self.feature2mask6 = FeatureToMask(1024)

        self.mask6_up = Up(1024, 512)
        self.mask5_up = Up(512, 256)
        self.mask4_up = Up(256, 128)
        self.mask3_up = Up(128, 64)
        self.mask2_up = Up(64, n_classes)

        self.mask_feature5 = MaskFeature(512)
        self.mask_feature4 = MaskFeature(256)
        self.mask_feature3 = MaskFeature(128)
        self.mask_feature2 = MaskFeature(64)
        self.mask_feature1 = MaskFeature(n_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        feature1 = self.inc(x)           # 32   x 512 x 512   [-1, 1]
        feature2 = self.down1(feature1)  # 64   x 256 x 256   [-1, 1]
        feature3 = self.down2(feature2)  # 128  x 128 x 128   [-1, 1]
        feature4 = self.down3(feature3)  # 256  x 64  x 64    [-1, 1]
        feature5 = self.down4(feature4)  # 512  x 32  x 32    [-1, 1]
        feature6 = self.down5(feature5)  # 1024 x 16  x 16    [-1, 1]

        mask6 = self.feature2mask6(feature6)                 # [0, 1]
        mask6_up = self.mask6_up(mask6)
        mask5 = self.mask_feature5(feature5, mask6_up)
        mask5_up = self.mask5_up(mask5)
        mask4 = self.mask_feature4(feature4, mask5_up)
        mask4_up = self.mask4_up(mask4)
        mask3 = self.mask_feature3(feature3, mask4_up)
        mask3_up = self.mask3_up(mask3)
        mask2 = self.mask_feature2(feature2, mask3_up)
        mask2_up = self.mask2_up(mask2)
        mask_out = self.mask_feature1(feature1, mask2_up)

        return mask_out

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
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
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


class FeatureToMask(nn.Module):
    def __init__(self, in_ch):
        super(FeatureToMask, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x


class MaskFeature(nn.Module):
    def __init__(self, in_ch):
        super(MaskFeature, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, feature, mask):
        x = self.conv(feature * mask)
        x = torch.sigmoid(x)
        return x
