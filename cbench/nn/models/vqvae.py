from turtle import forward
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions.utils import clamp_probs, broadcast_all
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
# from torch.utils.data import DataLoader
# import torch.optim as optim

from cbench.nn.distributions.relaxed import RelaxedOneHotCategorical, AsymptoticRelaxedOneHotCategorical, DoubleRelaxedOneHotCategorical

import math

from ..base import NNTrainableModule


# from https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
class VectorQuantizer(NNTrainableModule):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        loss = dict(
            q_latent_loss = q_latent_loss,
            e_latent_loss = self._commitment_cost * e_latent_loss,
        )
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous() #, perplexity, encodings


class VectorQuantizerEMA(NNTrainableModule):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # loss = self._commitment_cost * e_latent_loss
        loss = dict(e_latent_loss = self._commitment_cost * e_latent_loss)

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous() #, perplexity, encodings

# https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py
class VQEmbeddingEMA(NNTrainableModule):
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
        # loss = self.commitment_cost * e_latent_loss
        loss = dict(e_latent_loss = self.commitment_cost * e_latent_loss)

        if not self.training:
            self.update_cache("hist_dict",
                code_hist=indices.float().cpu().detach_()
            )

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        self.update_cache("metric_dict", 
            perplexity=perplexity.sum()
        )

        return loss, quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W) #, perplexity.sum()

# gumbel-softmax training vq from https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py
class VQEmbeddingGSSoft(NNTrainableModule):
    def __init__(self, latent_dim, num_embeddings, embedding_dim,
                 dist_type="RelaxedOneHotCategorical", # RelaxedOneHotCategorical, DoubleRelaxedOneHotCategorical
                 relax_temp=1.0, relax_temp_min=1.0, relax_temp_anneal=False, relax_temp_anneal_rate=1e-6,
                 kl_cost=1.0, use_st_gumbel=False, commitment_cost=0.0, commitment_over_exp=False,
                 test_sampling=False,
                 gs_temp=0.5, gs_temp_min=0.5, gs_anneal=False, gs_anneal_rate=1e-6,
        ):
        super(VQEmbeddingGSSoft, self).__init__()

        self.embedding = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
        # nn.init.uniform_(self.embedding, -1, 1)
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)
        
        self.dist_type = dist_type
        
        self.relax_temp_anneal = relax_temp_anneal
        self.relax_temp_anneal_rate = relax_temp_anneal_rate
        self.register_buffer("relax_temp", torch.tensor(max(relax_temp, relax_temp_min), requires_grad=False))
        self.register_buffer("relax_temp_min", torch.tensor(relax_temp_min, requires_grad=False))
        
        self.kl_cost = kl_cost
        self.use_st_gumbel = use_st_gumbel
        self.commitment_cost = commitment_cost
        self.commitment_over_exp = commitment_over_exp

        self.test_sampling = test_sampling

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

        if self.dist_type == "RelaxedOneHotCategorical":
            logits_norm = -distances - torch.logsumexp(-distances, dim=-1, keepdim=True) 
            dist = RelaxedOneHotCategorical(self.gs_temp,
                logits=(logits_norm / self.relax_temp),
            )
        elif self.dist_type == "AsymptoticRelaxedOneHotCategorical":
            dist = AsymptoticRelaxedOneHotCategorical(self.gs_temp,
                logits=(-distances),
            )
        elif self.dist_type == "DoubleRelaxedOneHotCategorical":
            dist = DoubleRelaxedOneHotCategorical(self.gs_temp, self.relax_temp,
                logits=(-distances),
            )
        else:
            raise ValueError(f"Unknown dist_type {self.dist_type} !")
        
        if self.training or self.test_sampling:
            samples = dist.rsample().view(N, -1, M)
            if self.use_st_gumbel:
                _, ind = samples.max(dim=-1)
                samples_hard = torch.zeros_like(samples).view(N, -1, M)
                samples_hard.scatter_(-1, ind.view(N, -1, 1), 1)
                samples_hard = samples_hard.view(N, -1, M)
                samples = samples_hard - samples.detach() + samples
            if not self.training:
                _, ind = samples.max(dim=-1)
                self.update_cache("hist_dict",
                    code_hist=ind.view(N, -1).float().cpu().detach_()
                )
                samples = torch.zeros_like(samples).view(N, -1, M)
                samples.scatter_(-1, ind.view(N, -1, 1), 1)
        else:
            samples = torch.argmax(dist.probs, dim=-1)
            if not self.training:
                self.update_cache("hist_dict",
                    code_hist=samples.view(N, -1).float().cpu().detach_()
                )
            samples = F.one_hot(samples, M).float()
            samples = samples.view(N, -1, M)
        
        quantized = torch.bmm(samples, self.embedding)
        quantized = quantized.view_as(x)

        KL = dist.probs * (dist.logits + math.log(M))
        KL[(dist.probs == 0).expand_as(KL)] = 0
        KL = KL.sum(dim=(0, 2, 3, 4)).mean()

        loss = dict(loss_rate=KL * self.kl_cost)

        if self.commitment_over_exp:
            e_latent_loss = (dist.probs * distances).mean()
        else:
            e_latent_loss = F.mse_loss(x, quantized.detach())
        # loss = self.commitment_cost * e_latent_loss
        loss.update(e_latent_loss = self.commitment_cost * e_latent_loss)


        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        self.update_cache("metric_dict", 
            perplexity=perplexity.sum()
        )

        # annealing
        # TODO: this may perform differently under distributed training!
        # use global parameter schedule if possible!
        if self.gs_anneal:
            if self.training:
                self.gs_temp = torch.maximum(self.gs_temp * math.exp(-self.gs_anneal_rate * B), self.gs_temp_min)
            self.update_cache("metric_dict", 
                gs_temp=self.gs_temp
            )
        if self.relax_temp_anneal:
            if self.training:
                self.relax_temp = torch.maximum(self.relax_temp * math.exp(-self.relax_temp_anneal_rate * B), self.relax_temp_min)
            self.update_cache("metric_dict", 
                relax_temp=self.relax_temp
            )

        return loss, quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W) #, perplexity.sum()

class PyramidVQEmbedding(NNTrainableModule):
    def __init__(self, latent_dim : int, pyramid_num_embeddings : List[int], embedding_dim : int,
        gs_temp=0.5, gs_temp_min=0.5, gs_anneal=False, gs_anneal_rate=1e-6,
        use_gssoft=False, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super().__init__()
        self.latent_dim = latent_dim
        # self.pyramid_num_embeddings = pyramid_num_embeddings
        self.embedding_dim = embedding_dim
        self.use_gssoft = use_gssoft

        self.gs_anneal = gs_anneal
        self.gs_anneal_rate = gs_anneal_rate
        # if gs_anneal:
        self.register_buffer("gs_temp", torch.tensor(max(gs_temp, gs_temp_min), requires_grad=False))
        self.register_buffer("gs_temp_min", torch.tensor(gs_temp_min, requires_grad=False))
        # else:
        #     self.gs_temp = gs_temp

        self.register_buffer('pyramid_num_embeddings', torch.as_tensor(pyramid_num_embeddings))

        if use_gssoft:
            vqs = [
                VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim)
                for num_embeddings in pyramid_num_embeddings
            ]
        else:
            vqs = [
                VQEmbeddingEMA(latent_dim, num_embeddings, embedding_dim, 
                    commitment_cost=commitment_cost,
                    decay=decay,
                    epsilon=epsilon,
                )
                for num_embeddings in pyramid_num_embeddings
            ]

        self.vqs = nn.ModuleList(vqs)

    def forward(self, x):
        B, C, H, W = x.size()
        N, NB, D = self.latent_dim, len(self.pyramid_num_embeddings), self.embedding_dim
        assert C == N * (D + NB)

        x, pyramid_level = torch.split(x, (N*D, N*NB), dim=1)
        # x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        # x_flat = x.reshape(N, -1, D)
        pyramid_level = pyramid_level.view(B, N, NB, H, W).permute(1, 0, 3, 4, 2)
        pyramid_level_flat = pyramid_level.reshape(N, -1, NB)

        # all quantize
        quantized_all = []
        kl_all = []
        for vq in self.vqs:
            KL, quantized = vq(x)
            quantized_all.append(quantized)
            kl_all.append(KL)

        # select level
        dist_level = distributions.RelaxedOneHotCategorical(self.gs_temp, logits=pyramid_level_flat)
        if self.training:
            samples = dist_level.rsample().reshape(N, -1, NB)
        else:
            samples = torch.argmax(dist_level.probs, dim=-1)
            samples = F.one_hot(samples, NB).float()
            samples = samples.reshape(N, -1, NB)

        quantized_all = torch.stack(quantized_all, dim=-1).view(B, N, D, H, W, NB).permute(1, 0, 3, 4, 5, 2).reshape(-1, NB, D)
        quantized_all = torch.bmm(samples.reshape(-1, 1, NB), quantized_all)
        quantized = quantized_all.view(N, B, H, W, D)
        # kl_all = torch.stack(kl_all, dim=-1)
        # KL = (samples * kl_all).sum(dim=-1).view(N, B, H, W)
        if self.use_gssoft:
            kl_all = torch.stack(kl_all, dim=-1)
        else:
            kl_all = torch.log(self.pyramid_num_embeddings.float())
        KL = dist_level.probs * (dist_level.logits + kl_all + math.log(NB))
        KL[(dist_level.probs == 0).expand_as(KL)] = 0
        KL = KL.view(N, B, H, W, NB).sum(dim=(0, 2, 3, 4)).mean()

        # gs annealing
        # TODO: this may perform differently under distributed training!
        # use global parameter schedule if possible!
        if self.training and self.gs_anneal:
            self.gs_temp = torch.maximum(self.gs_temp * math.exp(-self.gs_anneal_rate * B), self.gs_temp_min)
            self.update_cache("metric_dict", 
                gs_temp=self.gs_temp
            )

        return KL, quantized.permute(1, 0, 4, 2, 3).reshape(B, N * D, H, W) #, perplexity.sum()


class PyramidVQEmbeddingGSSoft(NNTrainableModule):
    def __init__(self, latent_dim : int, pyramid_num_embeddings : List[int], embedding_dim : int):
        super().__init__()

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
            for num_embeddings in pyramid_num_embeddings
        ])
        for idx, num_embeddings in enumerate(pyramid_num_embeddings):
            nn.init.uniform_(self.codebooks[idx], -1, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        N, _, D = self.codebooks[0].size()
        NB = len(self.codebooks)
        assert C == N * (D + NB)

        x, pyramid_level = torch.split(x, (N*D, N*NB), dim=1)
        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.reshape(N, -1, D)
        pyramid_level = pyramid_level.view(B, N, NB, H, W).permute(1, 0, 3, 4, 2)
        pyramid_level_flat = pyramid_level.reshape(N, -1, NB)

        # all quantize
        quantized_all = []
        kl_all = []
        for embedding in self.codebooks:
            M = embedding.size(1)

            distances = torch.baddbmm(torch.sum(embedding ** 2, dim=2).unsqueeze(1) +
                                    torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                    x_flat, embedding.transpose(1, 2),
                                    alpha=-2.0, beta=1.0)
            distances = distances.view(N, B, H, W, M)

            dist = distributions.RelaxedOneHotCategorical(0.5, logits=-distances)
            if self.training:
                samples = dist.rsample().view(N, -1, M)
            else:
                samples = torch.argmax(dist.probs, dim=-1)
                samples = F.one_hot(samples, M).float()
                samples = samples.view(N, -1, M)

            quantized = torch.bmm(samples, embedding)
            # quantized = quantized.view_as(x)
            quantized_all.append(quantized)

            KL = dist.probs * (dist.logits + math.log(M)) # aka -p * log(Mp)
            KL[(dist.probs == 0).expand_as(KL)] = 0
            # KL = KL.sum(dim=(0, 2, 3, 4)).mean()
            KL = KL.view(N, -1, M).sum(dim=-1)
            kl_all.append(KL)

            # avg_probs = torch.mean(samples, dim=1)
            # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        # select level

        dist_level = distributions.RelaxedOneHotCategorical(0.5, logits=pyramid_level_flat)
        if self.training:
            samples = dist_level.rsample().view(N, -1, NB)
        else:
            samples = torch.argmax(dist_level.probs, dim=-1)
            samples = F.one_hot(samples, NB).float()
            samples = samples.view(N, -1, NB)

        quantized_all = torch.stack(quantized_all, dim=-2).view(-1, NB, D)
        quantized_all = torch.bmm(samples.view(-1, 1, NB), quantized_all)
        quantized = quantized_all.view_as(x)
        kl_all = torch.stack(kl_all, dim=-1)
        KL = (samples * kl_all).sum(dim=-1).view(N, B, H, W)
        KL = KL.sum(dim=(0, 2, 3)).mean()

        return KL, quantized.permute(1, 0, 4, 2, 3).reshape(B, N * D, H, W) #, perplexity.sum()


class MultiVectorQuantizerWrapper(NNTrainableModule):
    def __init__(self, vqs : List[nn.Module]) -> None:
        super().__init__()

        self.vqs = nn.ModuleList(vqs)

    def forward(self, x):
        quantized_all = []
        loss_all = 0
        for vq in self.vqs:
            loss, quantized = vq(x)
            loss_all += loss
            quantized_all.append(quantized)

        loss_all /= len(self.vqs) # normalize
        # quantized_all = torch.stack(quantized_all)

        return loss_all, quantized_all
            


class Residual(NNTrainableModule):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens,
        use_batch_norm=False):
        super(Residual, self).__init__()
        if use_batch_norm:
            self._block = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels,
                        out_channels=num_residual_hiddens,
                        kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_residual_hiddens),
                nn.ReLU(True),
                nn.Conv2d(in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_hiddens),
            )
        else:
            self._block = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels,
                        out_channels=num_residual_hiddens,
                        kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1, stride=1, bias=False)
            )
    
    def forward(self, x):
        return x + self._block(x)


class MultiVectorQuantizerWrapper(NNTrainableModule):
    def __init__(self, vqs : List[nn.Module]) -> None:
        super().__init__()

        self.vqs = nn.ModuleList(vqs)

    def forward(self, x):
        quantized_all = []
        loss_all = 0
        for vq in self.vqs:
            loss, quantized = vq(x)
            loss_all += loss
            quantized_all.append(quantized)

        loss_all /= len(self.vqs) # normalize
        # quantized_all = torch.stack(quantized_all)

        return loss_all, quantized_all
            


class Residual(NNTrainableModule):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens,
        use_batch_norm=False):
        super(Residual, self).__init__()
        if use_batch_norm:
            self._block = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels,
                        out_channels=num_residual_hiddens,
                        kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_residual_hiddens),
                nn.ReLU(True),
                nn.Conv2d(in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_hiddens),
            )
        else:
            self._block = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels,
                        out_channels=num_residual_hiddens,
                        kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1, stride=1, bias=False)
            )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(NNTrainableModule):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
        use_batch_norm=False):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens, use_batch_norm=use_batch_norm)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class VQVAEEncoder(NNTrainableModule):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
        use_batch_norm=False):
        super(VQVAEEncoder, self).__init__()

        self.use_batch_norm = use_batch_norm

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        if use_batch_norm:
            self._bn_1 = nn.BatchNorm2d(num_hiddens//2)
            self._bn_2 = nn.BatchNorm2d(num_hiddens)
            self._bn_3 = nn.BatchNorm2d(num_hiddens)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens,
                                             use_batch_norm=use_batch_norm,)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        if self.use_batch_norm:
            x = self._bn_1(x)
        x = F.relu(x)
        
        x = self._conv_2(x)
        if self.use_batch_norm:
            x = self._bn_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        if self.use_batch_norm:
            x = self._bn_3(x)
        return self._residual_stack(x)

class VQVAEDecoder(NNTrainableModule):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
        use_batch_norm=False):
        super(VQVAEDecoder, self).__init__()
        self.use_batch_norm = use_batch_norm
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens,
                                             use_batch_norm=use_batch_norm,)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        if use_batch_norm:
            self._bn_1 = nn.BatchNorm2d(num_hiddens)
            self._bn_trans_1 = nn.BatchNorm2d(num_hiddens//2)
            # self._bn_trans_2 = nn.BatchNorm2d(num_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        if self.use_batch_norm:
            x = self._bn_1(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        if self.use_batch_norm:
            x = self._bn_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)

# default training params
# batch_size = 256
# num_training_updates = 15000

# num_hiddens = 128
# num_residual_hiddens = 32
# num_residual_layers = 2

# embedding_dim = 64
# num_embeddings = 512

# commitment_cost = 0.25

# decay = 0.99

# learning_rate = 1e-3
class VQVAEModel(NNTrainableModule):
    def __init__(self, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, 
                 num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.99):
        super(VQVAEModel, self).__init__()
        
        self._encoder = VQVAEEncoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = VQVAEDecoder(embedding_dim, 3,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon #, perplexity