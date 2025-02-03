import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_weights=True):
        super(UNet, self).__init__()

        self.down1 = DonwSampleBlock(n_channels, 32)
        self.down2 = DonwSampleBlock(32, 32)
        self.down3 = DonwSampleBlock(32, 64)
        self.down4 = DonwSampleBlock(64, 64)
        self.down5 = DonwSampleBlock(64, 128)
        self.down6 = DonwSampleBlock(128, 128)
        self.down7 = DonwSampleBlock(128, 256, down_opt=False)

        self.fc = FCBlock(256, 1024)

        self.up7 = UpSampleBlock(256, 128, up_opt=False)
        self.up6 = UpSampleBlock(128, 128)
        self.up5 = UpSampleBlock(128, 64)
        self.up4 = UpSampleBlock(64, 64)
        self.up3 = UpSampleBlock(64, 32)
        self.up2 = UpSampleBlock(32, 32)
        self.up1 = UpSampleBlock(32, 32)

        self.out = OutBlock(32, n_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1_skip, x1_down = self.down1(x)
        x2_skip, x2_down = self.down2(x1_down)
        x3_skip, x3_down = self.down3(x2_down)
        x4_skip, x4_down = self.down4(x3_down)
        x5_skip, x5_down = self.down5(x4_down)
        x6_skip, x6_down = self.down6(x5_down)
        x7_skip = self.down7(x6_down)

        x_fc = self.fc(x7_skip)

        x7_up = self.up7(x_fc, x7_skip)
        x6_up = self.up6(x7_up, x6_skip)
        x5_up = self.up5(x6_up, x5_skip)
        x4_up = self.up4(x5_up, x4_skip)
        x3_up = self.up3(x4_up, x3_skip)
        x2_up = self.up2(x3_up, x2_skip)
        x1_up = self.up1(x2_up, x1_skip)
        x = self.out(x1_up)

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


class DonwSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down_opt=True):
        super(DonwSampleBlock, self).__init__()
        self.down_opt = down_opt
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0)
        )
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_res = self.res(x)
        x = x_conv + x_res
        if self.down_opt:
            x_down = self.pool(x)
            return x, x_down
        else:
            return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_opt=True):
        super(UpSampleBlock, self).__init__()
        self.up_opt = up_opt
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, padding=0)
        )

    def forward(self, x_up, x_skip):
        if self.up_opt:
            x_up = self.up(x_up)
        x = torch.cat((x_up, x_skip), dim=1)
        x_conv = self.conv(x)
        x_res = self.res(x)
        x = x_conv + x_res
        return x


class FCBlock(nn.Module):
    def __init__(self, in_ch, fc_num):
        super(FCBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, fc_num, 8, padding=0)
        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(fc_num, fc_num),
        )
        self.back = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(fc_num, in_ch*8*8),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x_fc = self.fc(x)
        x = x + x_fc
        x_fc = self.fc(x)
        x = x + x_fc
        x = self.back(x)
        x = torch.reshape(x, shape=[-1, 256, 8, 8])
        return x


class OutBlock(nn.Module):
    def __init__(self, in_ch, n_classes):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, 1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)