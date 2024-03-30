import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FCN7(nn.Module):
    def __init__(self):
        super(FCN7, self).__init__()
        self.conv1 = ConvBlock(1, 128)
        self.mp1 = nn.MaxPool2d((2, 4))
        
        self.conv2 = ConvBlock(128, 256)
        self.mp2 = nn.MaxPool2d((2, 4))
        
        self.conv3 = ConvBlock(256, 512)
        self.mp3 = nn.MaxPool2d((2, 4))
        
        self.conv4 = ConvBlock(512, 1024)
        self.mp4 = nn.MaxPool2d((3, 5))
        
        self.conv5 = ConvBlock(1024, 2048)
        self.mp5 = nn.MaxPool2d((4, 4))
        
        self.conv6 = nn.Conv2d(2048, 1024, kernel_size=1) # 1x1 convolutions
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1) # additional 1x1 convolution as per FCN-7
        
        # Fully connected layer with batch normalization and sigmoid activation
        self.fc = nn.Sequential(
            nn.Linear(512, class_num),
            nn.BatchNorm1d(class_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mp1(self.conv1(x))
        x = self.mp2(self.conv2(x))
        x = self.mp3(self.conv3(x))
        x = self.mp4(self.conv4(x))
        x = self.mp5(self.conv5(x))
        
        x = self.conv6(x)
        x = self.conv7(x)
        
        x = x.view(x.size(0), -1)  # Flatten the output

        # Apply the fully connected layer
        x = self.fc(x)
        return x

#class_num = xxx  # Change this to your number of classes
#fcn7 = FCN7(class_num=class_num)

# Example input tensor
# inputs = torch.randn(1, 1, 96, 1366)  # Batch size of 1
# output = fcn7(inputs)
