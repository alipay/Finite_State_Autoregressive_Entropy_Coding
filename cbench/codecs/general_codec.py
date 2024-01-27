import io
import torch.nn as nn

from typing import Any, List
from cbench.modules.base import TrainableModuleInterface
from cbench.nn.base import NNTrainableModule
from .base import BaseCodec, BaseTrainableCodec

from cbench.modules.preprocessor.base import Preprocessor
from cbench.modules.prior_model.base import PriorModel
from cbench.modules.context_model.base import ContextModel
from cbench.modules.entropy_coder.base import EntropyCoder

from cbench.utils.logging_utils import MetricLogger, SmoothedValue
from cbench.utils.bytes_ops import merge_bytes, split_merged_bytes


class GeneralCodec(BaseTrainableCodec):

    def __init__(self, *args,
                 preprocessor: Preprocessor = None,
                 prior_model: PriorModel = None,
                 context_model: ContextModel = None,
                 entropy_coder: EntropyCoder = None,
                 prior_first=False,
                 **kwargs):
        """_summary_

        Args:
            preprocessor (Preprocessor, optional): Preprocessor for data. Defaults to None.
            prior_model (PriorModel, optional): Prior model for extracting latent from original (or preprocessed) data. Defaults to None.
            context_model (ContextModel, optional): Context model for updating latent. Defaults to None.
            entropy_coder (EntropyCoder, optional): Lossless entropy coder for original (or preprocessed) data. Defaults to None.
            prior_first (bool, optional): Whether to execute prior model first. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.preprocessor = preprocessor
        self.prior_model = prior_model
        self.context_model = context_model
        self.entropy_coder = entropy_coder

        self.prior_first = prior_first

    def compress(self, data, *args, **kwargs):

        latent = None
        prior = None

        if self.prior_first:
            if self.prior_model:
                with self.profiler.start_time_profile("time_compress_prior_model"):
                    latent = self.prior_model.extract(data, *args, **kwargs)
                    prior = self.prior_model.predict(
                        data, *args, prior=latent, **kwargs)

        if self.preprocessor:
            with self.profiler.start_time_profile("time_compress_preprocessor"):
                data = self.preprocessor.preprocess(
                    data, *args, prior=prior, **kwargs)

        if not self.prior_first:
            if self.prior_model:
                with self.profiler.start_time_profile("time_compress_prior_model"):
                    latent = self.prior_model.extract(data, *args, **kwargs)
                    prior = self.prior_model.predict(
                        data, *args, prior=latent, **kwargs)

        if self.context_model:
            with self.profiler.start_time_profile("time_compress_context_model"):
                prior = self.context_model.run_compress(
                    data, *args, prior=prior, **kwargs)

        if self.entropy_coder:
            with self.profiler.start_time_profile("time_compress_entropy_coder"):
                data = self.entropy_coder.encode(
                    data, *args, prior=prior, **kwargs)

        if self.prior_model:
            if self.entropy_coder:
                if isinstance(data, bytes) and isinstance(latent, bytes):
                    return merge_bytes([latent, data], num_segments=2)
                else:
                    return latent, data
            else:
                return latent
        else:
            return data

    def decompress(self, data, *args, **kwargs):
        latent = None
        prior = None

        if self.prior_model:
            if self.entropy_coder:
                if isinstance(data, bytes):
                    latent, data = split_merged_bytes(data, num_segments=2)
                else:
                    latent, data = data
            else:
                latent = data

            with self.profiler.start_time_profile("time_decompress_prior_model"):
                prior = self.prior_model.predict(
                    data, *args, prior=latent, **kwargs)
                
            if not self.entropy_coder:
                return prior

        if self.entropy_coder:
            if self.context_model:
                with self.profiler.start_time_profile("time_decompress_context_entropy"):
                    if self.entropy_coder:
                        self.entropy_coder.set_stream(data)
                    for prior_current in self.context_model.run_decompress(data, *args, prior=prior, **kwargs):
                        if self.entropy_coder:
                            data = self.entropy_coder.decode_from_stream(
                                *args, prior=prior_current, **kwargs)
            else:
                with self.profiler.start_time_profile("time_decompress_entropy_coder"):
                    data = self.entropy_coder.decode(
                        data, *args, prior=prior, **kwargs)

        if self.preprocessor:
            with self.profiler.start_time_profile("time_decompress_preprocessor"):
                data = self.preprocessor.postprocess(
                    data, *args, prior=prior, **kwargs)

        return data

    def forward(self, data, *args, **kwargs) -> None:
        latent = None
        prior = None

        if self.prior_first:
            if self.prior_model:
                with self.profiler.start_time_profile("time_forward_prior_model"):
                    if isinstance(self.prior_model, nn.Module):
                        prior = self.prior_model(data, *args, **kwargs)
                    # else:
                    #     latent = self.prior_model.extract(data, *args, **kwargs)
                    #     prior = self.prior_model.predict(data, *args, prior=latent, **kwargs)

        if self.preprocessor:
            with self.profiler.start_time_profile("time_forward_preprocessor"):
                if isinstance(self.preprocessor, nn.Module):
                    data = self.preprocessor(
                        data, *args, prior=prior, **kwargs)
                # else:
                #     data = self.preprocessor.preprocess(data, *args, prior=prior, **kwargs)

        if not self.prior_first:
            if self.prior_model:
                with self.profiler.start_time_profile("time_forward_prior_model"):
                    if isinstance(self.prior_model, nn.Module):
                        prior = self.prior_model(data, *args, **kwargs)
                    # else:
                    #     latent = self.prior_model.extract(data, *args, **kwargs)
                    #     prior = self.prior_model.predict(data, *args, prior=latent, **kwargs)

        if self.context_model:
            with self.profiler.start_time_profile("time_forward_context_model"):
                if isinstance(self.context_model, nn.Module):
                    prior = self.context_model(
                        data, *args, prior=prior, **kwargs)
                # else:
                #     prior = self.context_model.run_compress(data, *args, prior=prior, **kwargs)

        if self.entropy_coder:
            with self.profiler.start_time_profile("time_forward_entropy_coder"):
                if isinstance(self.entropy_coder, nn.Module):
                    data = self.entropy_coder(
                        data, *args, prior=prior, **kwargs)
                # else:
                #     data = self.entropy_coder.encode(data, *args, prior=prior, **kwargs)

        if self.prior_model:
            if self.entropy_coder:
                return data
            else:
                return prior
        else:
            return data

    # TODO: implement those functions in a base class! (Like torch.nn.Module)
    # def get_parameters(self, *args, **kwargs) -> Dict[str, Any]:
    #     parameters = []
    #     for module in (self.preprocessor, self.prior_model, self.context_model, self.entropy_coder):
    #         if isinstance(module, TrainableModuleInterface):
    #             parameters.append(module.get_parameters(*args, **kwargs))
    #     return parameters

    # def load_parameters(self, parameters: Dict[str, Any], *args, **kwargs) -> None:
    #     idx = 0
    #     for module in (self.preprocessor, self.prior_model, self.context_model, self.entropy_coder):
    #         if isinstance(module, TrainableModuleInterface):
    #             module.load_parameters(parameters[idx], *args, **kwargs)
    #             idx += 1

    def train_full(self, dataloader, *args, **kwargs) -> None:
        data_all = list(dataloader)
        latent_all = [None] * len(data_all)
        prior_all = [None] * len(data_all)
        if self.prior_first:
            if self.prior_model:
                with self.profiler.start_time_profile("time_train_prior_model"):
                    if isinstance(self.prior_model, TrainableModuleInterface):
                        self.prior_model.train_full(data_all, *args, **kwargs)
                    latent_all = [self.prior_model.extract(
                        data, *args, **kwargs) for data in data_all]
                    prior_all = [self.prior_model.predict(
                        data, *args, prior=latent, **kwargs) for data, latent in zip(data_all, latent_all)]

        if self.preprocessor:
            with self.profiler.start_time_profile("time_train_preprocessor"):
                if isinstance(self.preprocessor, TrainableModuleInterface):
                    self.preprocessor.train_full(
                        data_all, *args, prior=prior_all, **kwargs)
                data_all = [self.preprocessor.preprocess(
                    data, *args, prior=prior, **kwargs) for data, prior in zip(data_all, prior_all)]

        if not self.prior_first:
            if self.prior_model:
                with self.profiler.start_time_profile("time_train_prior_model"):
                    if isinstance(self.prior_model, TrainableModuleInterface):
                        self.prior_model.train_full(data_all, *args, **kwargs)
                    latent_all = [self.prior_model.extract(
                        data, *args, **kwargs) for data in data_all]
                    prior_all = [self.prior_model.predict(
                        data, *args, prior=latent, **kwargs) for data, latent in zip(data_all, latent_all)]

        if self.context_model:
            with self.profiler.start_time_profile("time_train_context_model"):
                if isinstance(self.context_model, TrainableModuleInterface):
                    self.context_model.train_full(
                        data_all, *args, prior=prior_all, **kwargs)
                prior_all = [self.context_model.run_compress(
                    data, *args, prior=prior, **kwargs) for data, prior in zip(data_all, prior_all)]

        if self.entropy_coder:
            with self.profiler.start_time_profile("time_train_entropy_coder"):
                if isinstance(self.entropy_coder, TrainableModuleInterface):
                    self.entropy_coder.train_full(
                        data_all, *args, prior=prior_all, **kwargs)
                # this is the last training step, so no need to run anymore
                # data_all = [self.entropy_coder.encode(data, *args, prior=prior, **kwargs) for data, prior in zip(data_all, prior_all)]

    def train_iter(self, data, *args, **kwargs) -> None:
        latent = None
        prior = None

        if self.prior_first:
            if self.prior_model:
                with self.profiler.start_time_profile("time_train_prior_model"):
                    if isinstance(self.prior_model, TrainableModuleInterface):
                        self.prior_model.train_iter(data, *args, **kwargs)
                    latent = self.prior_model.extract(data, *args, **kwargs)
                    prior = self.prior_model.predict(
                        data, *args, prior=latent, **kwargs)

        if self.preprocessor:
            with self.profiler.start_time_profile("time_train_preprocessor"):
                if isinstance(self.preprocessor, TrainableModuleInterface):
                    self.preprocessor.train_iter(
                        data, *args, prior=prior, **kwargs)
                data = self.preprocessor.preprocess(
                    data, *args, prior=prior, **kwargs)

        if not self.prior_first:
            if self.prior_model:
                with self.profiler.start_time_profile("time_train_prior_model"):
                    if isinstance(self.prior_model, TrainableModuleInterface):
                        self.prior_model.train_iter(data, *args, **kwargs)
                    latent = self.prior_model.extract(data, *args, **kwargs)
                    prior = self.prior_model.predict(
                        data, *args, prior=latent, **kwargs)

        if self.context_model:
            with self.profiler.start_time_profile("time_train_context_model"):
                if isinstance(self.context_model, TrainableModuleInterface):
                    self.context_model.train_iter(
                        data, *args, prior=prior, **kwargs)
                prior = self.context_model.run_compress(
                    data, *args, prior=prior, **kwargs)

        if self.entropy_coder:
            with self.profiler.start_time_profile("time_train_entropy_coder"):
                if isinstance(self.entropy_coder, TrainableModuleInterface):
                    self.entropy_coder.train_iter(
                        data, *args, prior=prior, **kwargs)
                data = self.entropy_coder.encode(
                    data, *args, prior=prior, **kwargs)
