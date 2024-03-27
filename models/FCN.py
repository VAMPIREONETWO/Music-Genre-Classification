import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN7(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCN7, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = self._make_block(64, 128, 2)
        self.block2 = self._make_block(128, 256, 2)
        self.block3 = self._make_block(256, 512, 2)
        self.block4 = self._make_block(512, 1024, 2)
        self.block5 = self._make_block(1024, 2048, 2)
        self.block6 = nn.Conv2d(2048, 4096, kernel_size=1)
        self.block7 = nn.Conv2d(4096, num_classes, kernel_size=1)

    def _make_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = F.relu(self.block6(x))
        x = self.block7(x)
        
        # Global Average Pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        
        return x


# model = FCN7(in_channels=1, num_classes=50)
