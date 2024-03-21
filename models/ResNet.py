from torch.nn import (Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d,
                      Linear, AdaptiveAvgPool2d, BatchNorm1d, Sigmoid, Dropout)


class ResNet18(Module):
    def __init__(self, class_num, pre_filter_size=7, in_channels=3):
        super(ResNet18, self).__init__()

        # preprocessing layer
        self.pl = Sequential(
            Conv2d(in_channels, 64, kernel_size=pre_filter_size, stride=2, padding=2),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        # Residual Blocks
        self.block1 = ResidualBlock(64, 64)
        self.block2 = ResidualBlock(64, 64, dropout=True)
        self.block3 = ResidualBlock(64, 128, 2)
        self.block4 = ResidualBlock(128, 128, dropout=True)
        self.block5 = ResidualBlock(128, 256, 2)
        self.block6 = ResidualBlock(256, 256, dropout=True)
        self.block7 = ResidualBlock(256, 512, 2)
        self.block8 = ResidualBlock(512, 512)

        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Sequential(Linear(512, class_num),
                             BatchNorm1d(class_num),
                             Sigmoid())

    def forward(self, x):
        o = self.pl(x)
        o = self.block1(o)
        o = self.block2(o)
        o = self.block3(o)
        o = self.block4(o)
        o = self.block5(o)
        o = self.block6(o)
        o = self.block7(o)
        o = self.block8(o)
        o = self.avg_pool(o)
        o = o.view(o.size(0), -1)
        o = self.fc(o)
        return o


class ResidualBlock(Module):
    def __init__(self, in_filters, out_filters, strides=1, dropout=False):
        super().__init__()

        self.cl1 = Conv2d(in_filters, out_filters, kernel_size=3, stride=strides, padding=1)
        self.bn1 = BatchNorm2d(out_filters)
        self.relu1 = ReLU()

        self.cl2 = Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(out_filters)

        if strides != 1:
            self.shortcut = Sequential(Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=strides),
                                       BatchNorm2d(out_filters))
        else:
            self.shortcut = lambda x: x
        self.dropout = dropout
        if dropout:
            self.dp = Dropout(0.5)

    def forward(self, x):
        o = self.cl1(x)
        o = self.bn1(o)
        o = self.relu1(o)
        o = self.cl2(o)
        o = self.bn2(o)
        o = o + self.shortcut(x)
        if self.dropout:
            o = self.dp(o)
        return o
