import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.distributions import Normal, Bernoulli, kl_divergence
from torchvision.utils import save_image

import numpy as np
from autograd.builtins import tuple as ag_tuple
import craystack as cs
# import craystack.bb_ans as bb_ans
import struct

from .base import EntropyCoder, TorchQuantizedEntropyCoder

from cbench.nn.base import NNTrainableModule
from cbench.nn.models.vqvae_model_v2 import Encoder, Decoder
from cbench.nn.utils import batched_cross_entropy

from craystack.codecs import substack, Uniform, \
    std_gaussian_centres, DiagGaussian_StdBins, Codec




class BitsBackANSCoder(TorchQuantizedEntropyCoder, NNTrainableModule):
    def __init__(self, encoder, decoder, *args, 
        in_channels=3, out_channels=768, hidden_channels=256,
        obs_codec_type="categorical",
        fixed_batch_size=None,
        fixed_spatial_shape=None,
        prior_precision=8,
        obs_precision=16,
        q_precision=14,
        **kwargs):
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = hidden_channels // 2
        self.fixed_batch_size = fixed_batch_size
        self.fixed_spatial_shape = fixed_spatial_shape

        self.encoder = encoder
        self.decoder = decoder

        self.obs_codec_type = obs_codec_type
        if self.obs_codec_type == "categorical":
            obs_codec = lambda p: cs.Categorical(torch.softmax(p.reshape(p.shape[0], self.in_channels, p.shape[1] // self.in_channels, *p.shape[2:]).movedim(2, -1), dim=-1).detach().cpu().numpy(), obs_precision)
        elif self.obs_codec_type == "gaussian":
            def _obs_gaussian_codec(p):
                mean, logvar = p.chunk(2, dim=1)
                mean, stdd = mean.detach().cpu().numpy(), torch.exp(0.5 * logvar).detach().cpu().numpy()
                return cs.DiagGaussian_UnifBins(mean, stdd, self.data_range[0], self.data_range[1], obs_precision, self.data_precision+1)
            obs_codec = _obs_gaussian_codec
        else:
            raise NotImplementedError(f"Unknown obs_codec_type {obs_codec_type}")

        latent_shape = (4, 128, 8, 8)
        latent_size = np.prod(latent_shape)
        obs_shape = (4, 3, 32, 32)
        obs_size = np.prod(obs_shape)

        self.initial_message_size = latent_size + obs_size
        def vae_view(head):
            # print(head)
            return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                            np.reshape(head[latent_size:], obs_shape)))

        def rec_net(input):
            with self.profiler.start_time_profile("time_encoder"):
                with torch.no_grad():
                    input = self._data_postprocess(input)
                    input = torch.as_tensor(input, dtype=torch.float32, device=self.device)
                    latent = self.encoder(input)
                    mean, logvar = latent.chunk(2, dim=1)
                    return mean.detach().cpu().numpy(), torch.exp(0.5 * logvar).detach().cpu().numpy()

        def gen_net(input):
            with self.profiler.start_time_profile("time_decoder"):
                with torch.no_grad():
                    obs = self.decoder(torch.as_tensor(input, dtype=torch.float32, device=self.device))
                    # obs = obs.reshape(obs.shape[0], self.in_channels, obs.shape[1] // self.in_channels, *obs.shape[2:]).movedim(2, -1)
                    # obs = torch.softmax(obs, dim=-1)
                    return obs #.detach().cpu().numpy()

        def BBANS(prior, likelihood, posterior):
            """
            This codec is for data modelled with a latent variable model as described
            in the paper 'Practical Lossless Compression with Latent Variable Models'
            currently under review for ICLR '19.

            latent        observed
            variable         data

                ( z ) ------> ( x )

            This assumes data x is modelled via a model which includes a latent
            variable. The model has a prior p(z), likelihood p(x | z) and (possibly
            approximate) posterior q(z | x). See the paper for more details.
            """
            prior_push, prior_pop = prior

            def push(message, data):
                with self.profiler.start_time_profile("time_compress_encoder"):
                    _, posterior_pop = posterior(data)
                with self.profiler.start_time_profile("time_compress_ans_decode"):
                    message, latent = posterior_pop(message)
                with self.profiler.start_time_profile("time_compress_decoder"):
                    likelihood_push, _ = likelihood(latent)
                with self.profiler.start_time_profile("time_compress_ans_encode_data"):
                    message, = likelihood_push(message, data)
                with self.profiler.start_time_profile("time_compress_ans_encode_latent"):
                    message, = prior_push(message, latent)
                return message,

            def pop(message):
                with self.profiler.start_time_profile("time_decompress_ans_decode_latent"):
                    message, latent = prior_pop(message)
                with self.profiler.start_time_profile("time_decompress_decoder"):
                    likelihood_pop = likelihood(latent).pop
                with self.profiler.start_time_profile("time_decompress_ans_decode_data"):
                    message, data = likelihood_pop(message)
                with self.profiler.start_time_profile("time_decompress_encoder"):
                    posterior_push = posterior(data).push
                with self.profiler.start_time_profile("time_decompress_ans_encode"):
                    message, = posterior_push(message, latent)
                return message, data
            return Codec(push, pop)

        def VAE(gen_net, rec_net, obs_codec, prior_prec, latent_prec):
            """
            This codec uses the BB-ANS algorithm to code data which is distributed
            according to a variational auto-encoder (VAE) model. It is assumed that the
            VAE uses an isotropic Gaussian prior and diagonal Gaussian for its
            posterior.
            """
            z_view = lambda head: head[0]
            x_view = lambda head: head[1]

            prior = substack(Uniform(prior_prec), z_view)

            def likelihood(latent_idxs):
                z = std_gaussian_centres(prior_prec)[latent_idxs]
                return substack(obs_codec(gen_net(z)), x_view)

            def posterior(data):
                post_mean, post_stdd = rec_net(data)
                return substack(DiagGaussian_StdBins(
                    post_mean, post_stdd, latent_prec, prior_prec), z_view)
            return BBANS(prior, likelihood, posterior)

        self.vae_codec = VAE(gen_net, rec_net, obs_codec, prior_precision, q_precision)

        self.vae_append, self.vae_pop = cs.substack(
            VAE(gen_net, rec_net, obs_codec, prior_precision, q_precision),
            vae_view)

        ## Load mnist images
        # images = datasets.MNIST(sys.argv[1], train=False, download=True).data.numpy()
        # images = np.uint64(rng.random_sample(np.shape(images)) < images / 255.)
        # images = np.split(np.reshape(images, (num_images, -1)), num_batches)
    
    @property
    def downsample_ratio(self):
        raise NotImplementedError()

    def _prepare_codec(self, data_shape):
        if self.fixed_batch_size is not None:
            batch_size = self.fixed_batch_size
            self.repeat_num = data_shape[0] // self.fixed_batch_size
        else:
            batch_size = data_shape[0]
            self.repeat_num = 1
        latent_shape = (batch_size, self.latent_channels) + tuple([dim // self.downsample_ratio for dim in data_shape[2:]])
        latent_size = np.prod(latent_shape)
        obs_shape = (batch_size, *data_shape[1:])
        obs_size = np.prod(obs_shape)

        self.initial_message_size = latent_size + obs_size
        def vae_view(head):
            # print(head)
            return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                            np.reshape(head[latent_size:], obs_shape)))
        
        self.vae_append, self.vae_pop = cs.repeat(
            cs.substack(self.vae_codec, vae_view), self.repeat_num
        )

    def encode(self, data, *args, prior=None, **kwargs) -> bytes:
        batch_size = data.shape[0]
        channel_size = data.shape[1]
        spatial_shape = data.shape[2:]

        self._prepare_codec(data.shape)

        ## Encode
        # Initialize message with some 'extra' bits
        # init_message = cs.base_message(self.initial_message_size, randomize=True)
        init_message = cs.random_message(self.initial_message_size, (self.initial_message_size,))

        data = self._data_preprocess(data)
        with self.profiler.start_time_profile("time_codec_encode"):
            message, = self.vae_append(init_message, np.split(data, self.repeat_num))

        flat_message = cs.flatten(message) # [len(init_message[0]):]

        # message_len = 32 * len(flat_message)
        # print("Used {} bits.".format(message_len))
        # print("This is {:.4f} bits per pixel.".format(message_len / num_pixels))
        byte_strings = []
        byte_strings.append(struct.pack("<H", batch_size))
        if self.fixed_spatial_shape is not None:
            assert spatial_shape == self.fixed_spatial_shape
        else:
            byte_strings.append(struct.pack("B", len(spatial_shape)))
            for dim in spatial_shape:
                byte_strings.append(struct.pack("<H", dim))

        byte_strings.append(flat_message.tobytes())

        return b''.join(byte_strings)

    def decode(self, byte_string: bytes, *args, prior=None, **kwargs):
        # if len(byte_string) == 0:
        #     return torch.zeros(1, self.latent_dim*self.categorical_dim, 8, 8, device=self.device)

        # decode shape from header
        byte_ptr = 0
        batch_dim = struct.unpack("<H", byte_string[byte_ptr:(byte_ptr+2)])[0]
        byte_ptr += 2

        if self.fixed_spatial_shape is not None:
            spatial_shape = self.fixed_spatial_shape
        else:
            num_shape_dims = struct.unpack("B", byte_string[byte_ptr:(byte_ptr+1)])[0]
            spatial_shape = []
            byte_ptr += 1
            for _ in range(num_shape_dims):
                spatial_shape.append(struct.unpack("<H", byte_string[byte_ptr:(byte_ptr+2)])[0])
                byte_ptr += 2
        # spatial_dim = np.prod(spatial_shape)

        self._prepare_codec((batch_dim, self.in_channels, *spatial_shape))

        ## Decode
        flat_message = np.frombuffer(byte_string[byte_ptr:], dtype=np.uint32)
        message = cs.unflatten(flat_message, self.initial_message_size)

        with self.profiler.start_time_profile("time_codec_decode"):
            message, images = self.vae_pop(message)
        images = np.concatenate(images)

        images = self._data_postprocess(images)

        return images

    def forward(self, input, *args, prior=None, **kwargs):
        latent = self.encoder(input)
        mean, logvar = latent.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)

        dist = Normal(mean, std)
        latent_rsample = dist.rsample()

        if prior is not None:
            prior_mean, prior_logvar = prior.chunk(2, dim=1)
            prior_std = torch.exp(0.5 * prior_logvar)
            prior_dist = Normal(prior_mean, prior_std)
            KLD = kl_divergence(dist, prior_dist).sum()
        else:
            KLD = torch.sum(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))
        
        if self.training:
            self.update_cache("loss_dict",
                loss_rate=KLD / input.numel(), # normalize by input size
            )

        self.update_cache("metric_dict",
            prior_entropy = KLD / input.numel(), 
        )

        output = self.decoder(latent_rsample)
        if self.obs_codec_type == "categorical":
            loss_distortion = batched_cross_entropy(output, input).sum() / input.numel()
        elif self.obs_codec_type == "gaussian":
            mean, logvar = output.chunk(2, dim=1)
            # mse = (mean - input).pow(2) 
            # loss_distortion = (mse / (2*logvar.exp()) + logvar / 2).mean()
            dist = Normal(mean, torch.exp(logvar))
            probs = dist.cdf(input + self.data_step / 2) - dist.cdf(input - self.data_step / 2)
            loss_distortion = -(probs + 1e-7).log().mean()
        if self.training:
            self.update_cache("loss_dict",
                loss_distortion=loss_distortion,
            )
        self.update_cache("metric_dict",
            estimated_x_epd = loss_distortion,
        )
        self.update_cache("metric_dict",
            estimated_epd = (KLD / input.numel() + loss_distortion),
        )
        
        return output


class BitsBackANSCoderBackboneV2(BitsBackANSCoder):
    def __init__(self,
                 in_channels=3, out_channels=768, hidden_channels=256,
                 num_downsample_layers=2, 
                 upsample_method="conv", 
                 num_residual_layers=2,
                 use_skip_connection=False,
                 encoder_use_batch_norm=True,
                 decoder_use_batch_norm=True,
                 decoder_batch_norm_track=True,
                 **kwargs):
        self.num_downsample_layers = num_downsample_layers
        encoder = Encoder(hidden_channels, in_channels=in_channels, out_channels=hidden_channels,
            num_downsample_layers=num_downsample_layers, 
            num_residual_layers=num_residual_layers, use_skip_connection=use_skip_connection,
            use_batch_norm=encoder_use_batch_norm)
        decoder = Decoder(hidden_channels, in_channels=hidden_channels // 2, out_channels=out_channels,
            num_upsample_layers=num_downsample_layers, upsample_method=upsample_method,
            num_residual_layers=num_residual_layers, use_skip_connection=use_skip_connection,
            use_batch_norm=decoder_use_batch_norm ,batch_norm_track=decoder_batch_norm_track)

        super().__init__(encoder=encoder, decoder=decoder, 
            in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
            **kwargs
        )

    @property
    def downsample_ratio(self):
        return 2 ** self.num_downsample_layers

