import torch
import torch.nn as nn


batch_size = 128

leaky_relu_slope = 0.2


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            self._block(features_d * 1, features_d * 2, 4, 2, 1),  # 16 * 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8 * 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4 * 4
            # final conv layer. Takes 4x4 and converts into 1 channel. Single value representing if value is fake or real
            nn.Conv2d(
                features_d * 8, 1, kernel_size=4, stride=2, padding=0
            ),  # 1 * 1 output
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Defines block
        """

        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,  # Since we're using batchnorm
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Inpuit N x z_dim x 1 x 1
            self._block(z_dim, features_g * 16, 4, 2, 0),
            # N x f_g * 16 x 4 x 4
            # 4x4 after this block
            self._block(
                features_g * 16, features_g * 8, 4, 2, 1
            ),  # 8 x 8 after this layer
            self._block(
                features_g * 8, features_g * 4, 4, 2, 1
            ),  # 16 x 16 after this layer
            self._block(
                features_g * 4, features_g * 2, 4, 2, 1
            ),  # 16 x 16 after this layer
            nn.ConvTranspose2d(
                features_g * 2, img_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),  # makes output between -1 and 1. Normalised
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # Upsacpe. use nn.ConvTranspose2D. Does opposite of conv layer
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,  # Since we're using batchnorm2d
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  # use relu activation function
        )

    def forward(self, x):
        return self.gen(x)


# Initialise weights
def init_weights(model):
    # mean 0, sd =0.2
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    init_weights(disc)
    # print(disc(x).shape)

    assert disc(x).shape == (N, 1, 1, 1)
    # print(N)
    gen = Generator(z_dim, in_channels, 8)

    # Initialise weights
    init_weights(gen)

    # generate latent noise
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    # print(gen(z).shape, " = ", N, in_channels, H, W)


test()

# Training
