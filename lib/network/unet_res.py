import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_weights=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.dres0 = ResConv(32, 32)
        self.dres1 = ResConv(32, 64)
        self.dres2 = ResConv(64, 128)
        self.dres3 = ResConv(128, 256)
        self.dres4 = ResConv(256, 512)
        self.dres5 = ResConv(512, 1024)
        self.dres6 = ResConv(1024, 2048)

        self.ures0 = ResConv(32, 32)
        self.ures1 = ResConv(64, 64)
        self.ures2 = ResConv(128, 128)
        self.ures3 = ResConv(256, 256)
        self.ures4 = ResConv(512, 512)
        self.ures5 = ResConv(1024, 1024)

        self.up5 = Up(2048, 1024)
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.up0 = Up(64, 32)

        self.outc = OutConv(32, n_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x_inc = self.inc(x)
        x0 = self.dres0(x_inc)

        x1 = nn.MaxPool2d(2, 2)(x0)
        x1 = self.dres1(x1)

        x2 = nn.MaxPool2d(2, 2)(x1)
        x2 = self.dres2(x2)

        x3 = nn.MaxPool2d(2, 2)(x2)
        x3 = self.dres3(x3)

        x4 = nn.MaxPool2d(2, 2)(x3)
        x4 = self.dres4(x4)

        x5 = nn.MaxPool2d(2, 2)(x4)
        x5 = self.dres5(x5)

        x6 = nn.MaxPool2d(2, 2)(x5)
        x6 = self.dres6(x6)

        x5_up = self.up5(x6)
        x5_added = torch.add(x5, x5_up)
        x5 = self.ures5(x5_added)

        x4_up = self.up4(x5)
        x4_added = torch.add(x4, x4_up)
        x4 = self.ures4(x4_added)

        x3_up = self.up3(x4)
        x3_added = torch.add(x3, x3_up)
        x3 = self.ures3(x3_added)

        x2_up = self.up2(x3)
        x2_added = torch.add(x2, x2_up)
        x2 = self.ures2(x2_added)

        x1_up = self.up1(x2)
        x1_added = torch.add(x1, x1_up)
        x1 = self.ures1(x1_added)

        x0_up = self.up0(x1)
        x0_added = torch.add(x0, x0_up)
        x0 = self.ures0(x0_added)

        out_added = torch.add(x0, x_inc)
        x = self.outc(out_added)

        # return torch.sigmoid(x)
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


class ResConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        nn.Module.__init__(self)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        res_out = self.res_conv(x)
        conv_out = self.conv(x)
        x = torch.add(res_out, conv_out)
        x = nn.ReLU(inplace=True)(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_ch, out_ch),
            nn.Conv2d(out_ch, out_ch, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x