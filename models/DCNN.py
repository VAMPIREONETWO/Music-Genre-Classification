from torch import nn


class DCNN(nn.Module):
    def __init__(self, class_num):
        super(DCNN, self).__init__()

        # preprocessing layer
        self.pl = ConLayer(1, 128, strides=3, pooling=False)

        # intermediate layers
        self.cons128 = nn.Sequential()
        for _ in range(2):
            self.cons128.append(ConLayer(128, 128))
        self.cons256 = nn.Sequential(ConLayer(128, 256))
        for _ in range(5):
            self.cons256.append(ConLayer(256, 256))
        self.con512 = ConLayer(256, 512)
        self.con_dropout = nn.Sequential(ConLayer(512, 512, kernel_size=1, pooling=False, padding=0),
                                         nn.Dropout(0.5))

        # fully connected layer
        self.fc = nn.Sequential(nn.Linear(512, class_num),
                                nn.BatchNorm1d(class_num),
                                nn.Sigmoid())

    def forward(self, x):
        o = self.pl(x)
        o = self.cons128(o)
        o = self.cons256(o)
        o = self.con512(o)
        o = self.con_dropout(o)
        o = o.view(o.shape[0], -1)
        o = self.fc(o)
        return o


class ConLayer(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, strides=1, pooling=True,padding=1):
        super(ConLayer, self).__init__()

        self.con = nn.Conv1d(in_filters, out_filters, kernel_size=kernel_size, stride=strides,padding=padding)
        self.bn = nn.BatchNorm1d(out_filters)
        self.relu = nn.ReLU()
        self.pooling = pooling
        if self.pooling:
            self.pool = nn.MaxPool1d(3, stride=3)

    def forward(self, x):
        o = self.con(x)
        o = self.bn(o)
        o = self.relu(o)
        if self.pooling:
            o = self.pool(o)
        return o
