from torch import nn


class DCNN(nn.Module):
    def __init__(self, class_num):
        super(DCNN, self).__init__()

        # preprocessing layer
        self.pl = ConLayer(1, 128, strides=3, pooling=False)

        # intermediate layers
        self.cons128 = [ConLayer(128, 128) for _ in range(2)]
        self.cons256 = [ConLayer(128, 256) for _ in range(6)]
        self.con512 = ConLayer(256, 512)
        self.con_dropout = nn.Sequential(ConLayer(512, 512, kernel_size=1, pooling=False),
                                         nn.Dropout(0.5))

        # fully connected layer
        self.fc = nn.Linear(512, class_num)

    def forward(self, x):
        o = self.pl(x)
        for con in self.cons128:
            o = con(o)
        for con in self.cons256:
            o = con(o)
        o = self.con512(o)
        o = self.con_dropout(o)
        o.view(o.shape[0], -1)
        o = self.fc(o)
        return o


class ConLayer(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, strides=1, pooling=True):
        super(ConLayer, self).__init__()

        self.con = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=strides, padding=1)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()
        self.pooling = pooling
        if self.pooling:
            self.pool = nn.MaxPool2d(3, stride=3)

    def forward(self, x):
        o = self.con(x)
        o = self.bn(o)
        o = self.relu(o)
        if self.pooling:
            o = self.pool(o)
        return o
