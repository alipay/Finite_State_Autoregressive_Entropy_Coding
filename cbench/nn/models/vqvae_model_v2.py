from tkinter import X
import tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, RelaxedOneHotCategorical
from torch.utils.tensorboard import SummaryWriter
import math

from cbench.nn.base import PLNNTrainableModule


class VQEmbeddingEMA(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.zeros(latent_dim, num_embeddings, embedding_dim)
        embedding.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(latent_dim, num_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances, dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = torch.gather(self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=1)

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        return loss, quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W) #, perplexity.sum()


class VQEmbeddingGSSoft(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim, training_soft_samples=True,
            gs_temp=0.5, gs_temp_min=0.5, gs_anneal=False, gs_anneal_rate=1e-6,
        ):
        super(VQEmbeddingGSSoft, self).__init__()

        self.embedding = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)
        
        self.training_soft_samples = training_soft_samples

        self.gs_anneal = gs_anneal
        self.gs_anneal_rate = gs_anneal_rate
        self.register_buffer("gs_temp", torch.tensor(max(gs_temp, gs_temp_min), requires_grad=False))
        self.register_buffer("gs_temp_min", torch.tensor(gs_temp_min, requires_grad=False))

    def forward(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)
        distances = distances.view(N, B, H, W, M)
        # print(x)

        dist = RelaxedOneHotCategorical(self.gs_temp, logits=-distances)
        if self.training and self.training_soft_samples:
            samples = dist.rsample().view(N, -1, M)
        else:
            samples = torch.argmax(dist.probs, dim=-1)
            samples = F.one_hot(samples, M).float()
            samples = samples.view(N, -1, M)

        quantized = torch.bmm(samples, self.embedding)
        quantized = quantized.view_as(x)

        KL = dist.probs * (dist.logits + math.log(M))
        KL[(dist.probs == 0).expand_as(KL)] = 0
        KL = KL.sum(dim=(0, 2, 3, 4)).mean()

        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        
        # gs annealing
        # TODO: this may perform differently under distributed training!
        # use global parameter schedule if possible!
        if self.training and self.gs_anneal:
            self.gs_temp = torch.maximum(self.gs_temp * math.exp(-self.gs_anneal_rate * B), self.gs_temp_min)

        return KL, quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W) #, perplexity.sum()


class Residual(nn.Module):
    def __init__(self, channels, use_batch_norm=True):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity(),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, channels, 
        latent_dim=None, 
        embedding_dim=None, 
        in_channels=3, 
        out_channels=None, 
        num_downsample_layers=2, 
        num_residual_layers=2,
        use_skip_connection=False,
        use_batch_norm=True,
        input_shift=-0.5
        ):
        super(Encoder, self).__init__()
        self.use_skip_connection = use_skip_connection

        if out_channels is None:
            assert(latent_dim is not None and embedding_dim is not None)
            out_channels = latent_dim * embedding_dim

        layers = []
        if num_downsample_layers > 0:
            layers.extend([
                nn.Conv2d(in_channels, channels, 4, 2, 1, bias=False),
            ])
        else:
            layers.extend([
                nn.Conv2d(in_channels, channels, 1, bias=False),
            ])
        layers.append(nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity())

        for _ in range(num_downsample_layers-1):
            layers.extend([
                nn.ReLU(True),
                nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity(),
            ])

        if self.use_skip_connection:
            self.downsample = nn.Sequential(*layers)
            layers = []

        for _ in range(num_residual_layers):
            layers.append(Residual(channels, use_batch_norm=use_batch_norm))
        layers.append(nn.Conv2d(channels, out_channels, 1))

        if self.use_skip_connection:
            self.residual = nn.Sequential(*layers)
            layers = []
        else:
            self.encoder = nn.Sequential(*layers)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, channels, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     Residual(channels),
        #     Residual(channels),
        #     nn.Conv2d(channels, out_channels, 1)
        # )

        self.input_shift = input_shift

    def forward(self, x):
        if self.use_skip_connection:
            x = self.downsample(x + self.input_shift)
            x = x + self.residual(x)
        else:
            x = self.encoder(x + self.input_shift)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, 
        latent_dim=None, 
        embedding_dim=None, 
        in_channels=None, 
        out_channels=3*256,
        num_upsample_layers=2,
        upsample_method="conv",
        num_residual_layers=2,
        use_skip_connection=False,
        use_batch_norm=True,
        batch_norm_track=True,
        ):
        super(Decoder, self).__init__()
        self.use_skip_connection = use_skip_connection

        if in_channels is None:
            assert(latent_dim is not None and embedding_dim is not None)
            in_channels = latent_dim * embedding_dim

        layers = [
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, track_running_stats=batch_norm_track) if use_batch_norm else nn.Identity(),
        ]
        if self.use_skip_connection:
            self.head = nn.Sequential(*layers)
            layers = []

        for _ in range(num_residual_layers):
            layers.append(Residual(channels, use_batch_norm=use_batch_norm))

        if self.use_skip_connection:
            self.residual = nn.Sequential(*layers)
            layers = []

        for _ in range(num_upsample_layers):
            if upsample_method == "conv":
                layers.extend([
                    nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(channels, track_running_stats=batch_norm_track) if use_batch_norm else nn.Identity(),
                    nn.ReLU(True),
                ])
            elif upsample_method == "pixelshuffle":
                layers.extend([
                    nn.Conv2d(channels, channels * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.BatchNorm2d(channels, track_running_stats=batch_norm_track) if use_batch_norm else nn.Identity(),
                    nn.ReLU(True),
                ])
            else:
                raise NotImplementedError(f"{upsample_method} invalid!")
        layers.append(nn.Conv2d(channels, out_channels, 1))

        if self.use_skip_connection:
            self.upsample = nn.Sequential(*layers)
        else:
            self.decoder = nn.Sequential(*layers)

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(in_channels, channels, 1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     Residual(channels),
        #     Residual(channels),
        #     nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(channels, out_channels, 1)
        # )

    def forward(self, x):
        # x = self.decoder(x)
        # B, _, H, W = x.size()
        # x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        # dist = Categorical(logits=x)
        if self.use_skip_connection:
            x = self.head(x)
            x = x + self.residual(x)
            x = self.upsample(x)
        else:
            x = self.decoder(x)
        return x


class VQVAE(PLNNTrainableModule):
    def __init__(self, channels=256, latent_dim=8, num_embeddings=128, embedding_dim=32, 
        input_shift=-0.5,
        lr=5e-4,
        **kwargs):
        super(VQVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.encoder = Encoder(channels, latent_dim, embedding_dim, input_shift=input_shift)
        self.codebook = VQEmbeddingEMA(latent_dim, num_embeddings, embedding_dim)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

        self.lr = lr
        
        self.do_train()

    def forward(self, x):
        images = x
        x = self.encoder(x)
        vq_loss, z = self.codebook(x)
        x = self.decoder(z)
        targets = images * 255
        targets = targets.long()
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        logp = -F.cross_entropy(x.reshape(-1, 256), targets.reshape(-1), reduction='sum') / images.size(0)
        loss = - logp / images.numel() * images.size(0)  + vq_loss
        KL = (images.shape[-1] // 4) * (images.shape[-2] // 4) * self.latent_dim * math.log(self.num_embeddings)
        elbo = (KL - logp) / images.numel() * images.size(0)
        if self.training:
            self.loss_dict.update(
                loss=loss,
            )
        self.metric_dict.update(
            logp=logp,
            vq_loss=vq_loss,
            elbo=elbo,
        )
        return (z, x), loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        images = batch
        KL = (images.shape[-1] // 4) * (images.shape[-2] // 4) * self.latent_dim * math.log(self.num_embeddings)
        x = self.encoder(images)
        vq_loss, x = self.codebook(x)
        x = self.decoder(x)
        # targets = (images + 0.5) * 255
        targets = images * 255
        targets = targets.long()
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        logp = -F.cross_entropy(x.reshape(-1, 256), targets.reshape(-1), reduction='sum') / images.size(0)
        loss = - logp / images.numel() * images.size(0)  + vq_loss
        # print(KL, -logp, loss)
        elbo = (KL - logp) / images.numel() * images.size(0)
        bpd = elbo / math.log(2)

        self.log("training/loss", loss.item())
        self.log("training/elbo", elbo.item())
        self.log("training/bpd", bpd.item())
        self.log("training/vqloss", vq_loss.item())
        self.log("training/logp", logp.item())
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        KL = (images.shape[-1] // 4) * (images.shape[-2] // 4) * self.latent_dim * math.log(self.num_embeddings)
        x = self.encoder(images)
        vq_loss, x = self.codebook(x)
        x = self.decoder(x)
        # targets = (images + 0.5) * 255
        targets = images * 255
        targets = targets.long()
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        logp = -F.cross_entropy(x.reshape(-1, 256), targets.reshape(-1), reduction='sum') / images.size(0)
        elbo = (KL - logp) / images.numel() * images.size(0)
        bpd = elbo / math.log(2)
        
        self.log("validation/elbo", elbo.item())
        self.log("validation/bpd", bpd.item())

        # tensorboard_logger = self.logger.experiment
        # if isinstance(tensorboard_logger, SummaryWriter):
        #     tensorboard_logger.add_histogram()

        return elbo

    # def validation_epoch_end(self, outputs):
    #     samples = torch.argmax(dist.logits, dim=-1)
    #     grid = utils.make_grid(samples.float() / 255)
    #     writer.add_image("reconstructions", grid, epoch)

class GSSOFT(PLNNTrainableModule):
    def __init__(self, channels=256, latent_dim=8, num_embeddings=128, embedding_dim=32, 
        training_soft_samples=True,
        gs_temp=0.5, gs_temp_min=0.5, gs_anneal=False, gs_anneal_rate=1e-6,
        input_shift=-0.5, lr=5e-4,
        **kwargs):
        super(GSSOFT, self).__init__(**kwargs)
        self.encoder = Encoder(channels, latent_dim, embedding_dim, input_shift=input_shift)
        self.codebook = VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim, training_soft_samples=training_soft_samples,
            gs_temp=gs_temp, gs_temp_min=gs_temp_min, gs_anneal=gs_anneal, gs_anneal_rate=gs_anneal_rate)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

        self.lr = lr
        
        self.do_train()

    def forward(self, x):
        images = x
        x = self.encoder(x)
        KL, z = self.codebook(x)
        x = self.decoder(z)
        targets = images * 255
        targets = targets.long()
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        logp = -F.cross_entropy(x.reshape(-1, 256), targets.reshape(-1), reduction='sum') / images.size(0)
        elbo = (KL - logp) / images.numel() * images.size(0)

        if self.training:
            self.loss_dict.update(
                loss=elbo,
            )
        self.metric_dict.update(
            logp=logp,
            KL=KL,
            elbo=elbo,
        )
        return (z, x), elbo

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        images = batch
        x = self.encoder(images)
        KL, x = self.codebook(x)
        x = self.decoder(x)
        # targets = (images + 0.5) * 255
        targets = images * 255
        targets = targets.long()
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        logp = -F.cross_entropy(x.reshape(-1, 256), targets.reshape(-1), reduction='sum') / images.size(0)
        loss = (KL - logp) / images.numel() * images.size(0)
        # print(KL, -logp, loss)
        elbo = (KL - logp) / images.numel() * images.size(0)
        bpd = elbo / math.log(2)

        self.log("training/loss", loss.item())
        self.log("training/elbo", elbo.item())
        self.log("training/bpd", bpd.item())
        self.log("training/KL", KL.item())
        self.log("training/logp", logp.item())

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        x = self.encoder(images)
        KL, x = self.codebook(x)
        x = self.decoder(x)
        # targets = (images + 0.5) * 255
        targets = images * 255
        targets = targets.long()
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        logp = -F.cross_entropy(x.reshape(-1, 256), targets.reshape(-1), reduction='sum') / images.size(0)
        elbo = (KL - logp) / images.numel() * images.size(0)
        bpd = elbo / math.log(2)
        
        self.log("validation/elbo", elbo.item())
        self.log("validation/bpd", bpd.item())

        return elbo
