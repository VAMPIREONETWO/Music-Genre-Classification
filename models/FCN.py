from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, ConvTranspose2d

class FCN(Module):
    def __init__(self, in_channels, n_classes):
        super(FCN, self).__init__()

        # Encoder
        self.encoder1 = Sequential(
            Conv2d(in_channels, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2)
        )

        self.encoder2 = Sequential(
            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2)
        )

        self.encoder3 = Sequential(
            Conv2d(128, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(2)
        )

        # Bottleneck
        self.bottleneck = Sequential(
            Conv2d(256, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(2)
        )

        # Decoder
        self.decoder1 = Sequential(
            ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(256),
            ReLU(),
        )

        self.decoder2 = Sequential(
            ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(128),
            ReLU(),
        )

        self.decoder3 = Sequential(
            ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.classifier = Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        b = self.bottleneck(e3)
        d1 = self.decoder1(b) + e3
        d2 = self.decoder2(d1) + e2
        d3 = self.decoder3(d2) + e1
        return self.classifier(d3)
