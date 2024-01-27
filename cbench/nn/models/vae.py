from turtle import forward
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, RelaxedOneHotCategorical


class VAEEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                #  latent_dim: int,
                 hidden_dims: List = [32, 64, 128, 256],
                 **kwargs) -> None:
        super().__init__()

        # self.latent_dim = latent_dim

        modules = []
        # if hidden_dims is None:
        #     hidden_dims = [32, 64, 128, 256, 512]


        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

    def forward(self, input):
        return self.encoder(input)


class VAEDecoder(nn.Module):
    def __init__(self,
                 out_channels: int,
                #  latent_dim: int,
                 hidden_dims: List = [32, 64, 128, 256],
                 **kwargs) -> None:
        super().__init__()

        # Build Decoder
        modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims = [hidden_dims[i] for i in range(len(hidden_dims) - 1, -1, -1)]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        modules.append(nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= out_channels,
                                      kernel_size= 3, padding= 1),
                            # nn.Tanh()
                            )
        )
        
        self.decoder = nn.Sequential(*modules)

    def forward(self, input):
        return self.decoder(input)

