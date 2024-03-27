import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN7(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCN7, self).__init__()

        # Preprocessing layer
        self.pl = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Blocks
        self.block1 = ResidualBlock(64, 128)
        self.block2 = ResidualBlock(128, 256)
        self.block3 = ResidualBlock(256, 512, 2)
        self.block4 = ResidualBlock(512, 1024, 2, dropout=True)
        self.block5 = ResidualBlock(1024, 2048, 2)
        self.block6 = ResidualBlock(2048, 4096, 2)
        self.block7 = ResidualBlock(4096, num_classes, 1)  # Output channels = num_classes

    def forward(self, x):
        x = self.pl(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        # Global Average Pooling and classification
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=False):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.dropout = nn.Dropout(0.5) if dropout else None

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.dropout:
            out = self.dropout(out)

        out += identity
        out = self.relu(out)

        return out

# model = FCN7(in_channels=1, num_classes=50)
