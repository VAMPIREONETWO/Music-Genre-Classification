from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, ConvTranspose2d

class FCN(Module):
    def __init__(self, in_channels, class_num):
        super(CustomFCN, self).__init__()

        self.initial = Sequential(
            Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Define simple residual blocks without bottleneck layers
        self.resblock1 = self._make_layer(64, 64, 2)
        self.resblock2 = self._make_layer(64, 128, 2)
        self.resblock3 = self._make_layer(128, 256, 2)
        self.resblock4 = self._make_layer(256, 512, 2)

        # Upsampling layers
        self.up1 = ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Final classifier layer
        self.classifier = Conv2d(64, class_num, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        return Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.classifier(x)
        return x

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.downsample = Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
