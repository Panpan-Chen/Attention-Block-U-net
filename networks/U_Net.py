import torch
from torch import nn


class DoubleCov(nn.Module):
    def __init__(self, input, output):
        super(DoubleCov, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input, output, 3, padding=1),
            # nn.Dropout(0.5),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            nn.Conv2d(output, output, 3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input, output):
        super(UNet, self).__init__()

        self.conv1 = DoubleCov(input, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleCov(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleCov(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleCov(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleCov(512, 1024)
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
