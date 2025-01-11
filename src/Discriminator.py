from torch import nn


class Discriminator(nn.Module):
    def __init__(self, imgSize, channelSize):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=channelSize, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)


        # Img is downscaled 4 times
        self.mlp = nn.Linear(in_features=(512*imgSize*imgSize)/(1<<4), out_features=1)



    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = nn.functional.leaky_relu(x, 0.2)

        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        return x

class Discriminator_MNIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)



        # Img is downscaled 4 times
        self.mlp = nn.Linear(in_features=128*7*7, out_features=1)



    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x