from torch import nn
class Generator(nn.Module):

    def __init__(self, imgSize, channelSize):
        super().__init__()

        self.mlp = nn.Linear(in_features=100, out_features=512*(imgSize<<4)*(imgSize<<4))

        # Upscale 4 times
        self.deConv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deConv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deConv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deConv4 = nn.ConvTranspose2d(in_channels=64, out_channels=channelSize, kernel_size=4, stride=2, padding=1)

    def forward(self, x):

        x = self.mlp(x)
        x = x.view(x.size(0), 512, 4, 4)
        x = nn.functional.relu(x)
        x = self.deConv1(x)
        x = nn.functional.relu(x)
        x = self.deConv2(x)
        x = nn.functional.relu(x)
        x = self.deConv3(x)
        x = nn.functional.relu(x)
        x = self.deConv4(x)
        x = nn.functional.tanh(x)

        return x


class Generator_MNIST(nn.Module):

    def __init__(self):
        super().__init__()

        self.mlp = nn.Linear(in_features=100, out_features=128*7*7)

        # Upscale 2 times
        self.deConv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deConv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)


    def forward(self, x):

        x = self.mlp(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = nn.functional.relu(x)
        x = self.deConv1(x)
        x = nn.functional.relu(x)
        x = self.deConv2(x)
        x = nn.functional.tanh(x)

        return x


