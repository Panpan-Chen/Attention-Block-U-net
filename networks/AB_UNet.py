import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BottlenNeck(nn.Module):
    def __init__(self, in_, planes, stride=1, downsample=None):
        super(BottlenNeck, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(inplace=True),

        )
        self.ca = ChannelAttention(planes * 4)  # ChannelAttention
        self.sa = SpatialAttention()  # SpatialAttention
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if stride != 1 or planes * 4 != in_:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )

    def forward(self, x):
        residual = x
        out = self.layer(x)
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class DoubleCov1(nn.Module):
    def __init__(self, input, output):
        super(DoubleCov1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input, output, 3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = BottlenNeck(output, output)

        self.conv2 = nn.Sequential(
            nn.Conv2d(output * 4, output, 3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )
        # self.relu = nn.ReLU(inplace=True)
        # if output != input:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(input, output, kernel_size=1, stride=3, bias=False),
        #         nn.BatchNorm2d(output)
        #     )
        # self.downsample = downsample

    def forward(self, x):
        # residual = x
        # if self.downsample is not None:
        #     residual = self.downsample(x)
        x1 = self.conv1(x)
        x2 = self.bottleneck(x1)
        out = self.conv2(x2)
        # out += residual
        # out = self.relu(out)
        return out


class DoubleCov(nn.Module):
    def __init__(self, input, output):
        super(DoubleCov, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input, output, 3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            nn.Conv2d(output, output, 3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AB_UNet(nn.Module):
    def __init__(self, input, output):
        super(UNetRev, self).__init__()

        self.conv1 = DoubleCov(input, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleCov(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleCov1(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleCov1(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleCov1(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleCov(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleCov(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleCov(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleCov(128, 64)
        self.conv10 = nn.Conv2d(64, output, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up1 = self.up1(c5)
        merge1 = torch.cat([up1, c4], dim=1)
        c6 = self.conv6(merge1)
        up2 = self.up2(c6)
        merge2 = torch.cat([up2, c3], dim=1)
        c7 = self.conv7(merge2)
        up3 = self.up3(c7)
        merge3 = torch.cat([up3, c2], dim=1)
        c8 = self.conv8(merge3)
        up4 = self.up4(c8)
        merge4 = torch.cat([up4, c1], dim=1)
        c9 = self.conv9(merge4)
        c10 = self.conv10(c9)
        return torch.sigmoid(c10)
