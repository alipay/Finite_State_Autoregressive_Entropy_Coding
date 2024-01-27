from typing import List, Union
import os
import csv
import pickle
import copy
import hashlib

from configs.class_builder import ClassBuilder, ClassBuilderBase, ClassBuilderList, NamedParamBase

from cbench.benchmark.basic_benchmark import BasicLosslessCompressionBenchmark
from cbench.benchmark.base import BaseBenchmark
from cbench.codecs.base import CodecInterface
from cbench.data.base import DataLoaderInterface


class GroupedCodecBenchmarkBuilder(BaseBenchmark, ClassBuilderBase):
    def __init__(self, codec_group_builder: ClassBuilderList,
                #  dataloader: DataLoaderInterface,
                 benchmark_builder: ClassBuilder,
                 *args,
                 codec_name_length_limit=100,
                 codec_name_hash_length=8,
                 group_name="",
                 **kwargs):
        self.codec_group_builder = codec_group_builder
        self.benchmark_builder = benchmark_builder
        self.codec_name_length_limit = codec_name_length_limit
        self.codec_name_hash_length = codec_name_hash_length
        self.group_name = group_name

        super().__init__(None, None, *args, **kwargs)

        self.cached_metrics = []

    @property
    def name(self) -> str:
        return self.group_name + self.benchmark_builder.name

    @property
    def param(self):
        return self.benchmark_builder.param

    def build_class(self, *args, **kwargs) -> object:
        self.codec_names = [cb.name for cb in self.codec_group_builder]
        self.codec_group = [cb.build_class() for cb in self.codec_group_builder]
        self.benchmark = self.benchmark_builder.build_class(*args, **kwargs)
        assert(isinstance(self.benchmark, BaseBenchmark))
        self.setup_engine_from_copy(self.benchmark)
        return self

    def run_benchmark(self, *args, 
        ignore_exist_metrics=False,
        codecs_ignore_exist_metrics=False,
        **kwargs):

        # check if the benchmark has been run by checking metric file
        if not ignore_exist_metrics and os.path.exists(self.metric_raw_file):
            self.logger.warning("Metric file {} already exists! Skipping benchmark...".format(self.metric_raw_file))
            self.logger.warning("Specify ignore_exist_metrics=True if you want to restart the benchmark.")
            return

        self.cached_metrics = []
        metric_data_all = []
        hparams_all = []
        names_all = []
        for idx, codec in enumerate(self.codec_group):
            codec_build_name = self.codec_group_builder[idx].build_name()
            codec_name_full = self.codec_group_builder[idx].name
            codec_name = self.codec_group_builder[idx].get_name_under_limit(
                name_length_limit=self.codec_name_length_limit, 
                hash_length=self.codec_name_hash_length,
            )
            codec_hashtag = self.codec_group_builder[idx].get_hashtag(hash_length=self.codec_name_hash_length)
            # avoid filename too long 
            # if len(codec_name_full) > self.codec_name_length_limit:
            #     config_hash = hashlib.sha256(codec_name_full.encode()).hexdigest()[:self.codec_name_hash_length]
            #     codec_name = f"{config_hash}:{codec_name_full[:(self.codec_name_length_limit-len(config_hash))]}..."
            # else:
            #     codec_name = codec_name_full

            # setup benchmark
            codec_output_dir = os.path.join(self.output_dir, codec_name)
            assert(isinstance(self.benchmark, BaseBenchmark))
            self.benchmark.setup_engine(
                output_dir=codec_output_dir
            )
            self.benchmark.set_codec(codec)

            # save full codec name
            with open(os.path.join(codec_output_dir, "build_name.txt"), 'w') as f:
                f.write(codec_build_name)
            with open(os.path.join(codec_output_dir, "config_name.txt"), 'w') as f:
                f.write(codec_name_full)
            # TODO: use ClassBuilderDict to pass in a custom exp name for the codec
            with open(os.path.join(codec_output_dir, "exp_name.txt"), 'w') as f:
                f.write(codec_name_full)

            # save a copy of class builder in codec_output_dir for reproduction
            benchmark_builder_copy = copy.deepcopy(self.benchmark_builder)
            with open(os.path.join(codec_output_dir, 'config.pkl'), 'wb') as f: 
                pickle.dump(benchmark_builder_copy.update_args(self.codec_group_builder[idx]), f)
            
            self.logger.info(f"Running benchmark with codec: {codec_name_full}")
            self.logger.info(f"Output dir: {codec_output_dir}")
            # run benchmark and save metrics
            metric_dict = self.benchmark.run_benchmark(*args, 
                ignore_exist_metrics=codecs_ignore_exist_metrics, **kwargs)
            self.cached_metrics.append(metric_dict)
            config_dict = {name:param for name, param in self.codec_group_builder[idx].iter_slot_data()}
            # config_and_metric_dict = dict()
            # if isinstance(metric_dict, dict):
            #     config_and_metric_dict.update(**metric_dict)
            #     # config_and_metric_dict.update(codec_name=self.codec_group_builder[idx].name)
            # config_and_metric_dict.update(**config_dict)
            metric_data_all.append(metric_dict)
            hparams_all.append(config_dict)
            names_all.append(codec_name)
        self.save_metrics(metric_data=metric_data_all, hparams=hparams_all, names=names_all)

        return self.collect_metrics()

    def collect_metrics(self, *args, **kwargs):
        # TODO: draw a plot?
        return self.cached_metrics