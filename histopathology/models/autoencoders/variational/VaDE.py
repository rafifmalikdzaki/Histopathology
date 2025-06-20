import cupy as cp
import lightning as L
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import scipy as sp
from sklearn import mixture
from sklearn.cluster import KMeans
from torch.distributions import Normal, MultivariateNormal

import warnings
warnings.filterwarnings("ignore")


class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc == nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x*y.expand_as(x)


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 padding=1, dilation=1, bias=False):
        super().__init__()
        self.Convolution = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.Convolution(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = ConvolutionBlock(in_channels, out_channels)
        self.c2 = ConvolutionBlock(out_channels, out_channels)
        self.ai = SqueezeExcitation(out_channels)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.ai(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_c=3, latent_dim=64):
        super().__init__()
        self.dcs = DoubleConv(in_c, 256)
        self.lastConv = nn.Sequential(
            nn.Conv2d(256, 384),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True)
        )
        self.fc_mu = nn.Linear(384*4, latent_dim)
        self.fc_var = nn.Linear(384*4, latent_dim)

    def forward(self, x):
        result = self.dcs(x)
        result = self.lastConv(result)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return (mu, log_var)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.decodeInput = nn.Linear(latent_dim, 384*4)
        self.ConvTrans = nn.Sequential(
            nn.ConvTranspose2d(384, 256,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True)
        )
        self.DoubleConvTrans = nn.Sequential(
            nn.ConvTranspose2d(256, 256,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(256, 256,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True)
        )
        self.finalLayer = nn.Sequential(
            nn.ConvTranspose2d(256, 256,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, Z):
        result = self.decodeInput(Z).view(-1, 384, 2, 2)
        result = self.ConvTrans(result)
        result = self.DoubleConvTrans(result)
        result = self.finalLayer(result)
        return result


class VaDE(nn.Module):
    def __init__(self, in_c=3, latent_dim=64):
        self.latent_dim = latent_dim
        self.in_c = in_c
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def vae_loss(self, *args, **kwargs):
        reconstruction = args[0]
        input = args[1]
        z = args[2]
        mu = args[3]
        log_var = args[4]

        batch_size = input.size(0)
        pass

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples
