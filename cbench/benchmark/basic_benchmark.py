import csv
import pickle
import random
import time
import os
import json
import hashlib
import logging
import multiprocessing
import traceback
from tqdm import tqdm
from typing import Iterable, Sequence, Union, List, Tuple, Callable, Any, Optional
import numpy as np

# TODO: fix distributed overrun during testing to remove dependence on torch
import torch.distributed

from cbench.modules.base import TrainableModuleInterface

from .base import BaseBenchmark, BaseEngine
from .trainer import BasicTrainer
from .metrics.base import BaseMetric
from .metrics.bj_delta import BJDeltaMetric

from cbench.codecs.base import BaseCodec, CodecInterface
from cbench.data.dataloaders.basic import DataLoaderInterface
from cbench.utils.logging_utils import MetricLogger, SmoothedValue
from cbench.nn.base import NNTrainableModule

class _BenchmarkTestingWorker(object):
    def __init__(self, codec: CodecInterface, dataloader: DataLoaderInterface,
                #  metric_logger=None,
                 cache_dir=None,
                 cache_compressed_data=False,
                 cache_checksum=True,
                 skip_decompress=False,
                 **kwargs
        ) -> None:
        self.codec = codec
        self.dataloader = dataloader

        # self.metric_logger = metric_logger
        self.cache_dir = cache_dir
        self.cache_compressed_data = cache_compressed_data
        self.cache_checksum = cache_checksum
        self.skip_decompress = skip_decompress

    def __call__(self, idxs: Iterable[int] = None):

        metric_logger = MetricLogger()

        if idxs is None:
            idxs = range(len(self.dataloader))

        # for step, data in dataloader:
        for idx in tqdm(idxs, desc=f"pid={os.getpid()}"):
            step = idx
            if idx >= len(self.dataloader): continue
            data = self.dataloader[idx]

            # search cache
            compressed_data = None
            if self.cache_dir:
                if self.cache_compressed_data:
                    # NOTE: maybe using hash values and filenames? dataloader may be shuffled!
                    cache_file = os.path.join(self.cache_dir, "{}.bin".format(step))
                    if os.path.exists(cache_file):
                        with open(cache_file, 'rb') as f:
                            compressed_data = pickle.load(f)
                            if self.cache_checksum:
                                compressed_data, checksum = compressed_data
                                original_data_checksum = hashlib.md5(data).digest()
                                # checksum fail! rerun codec!
                                if checksum != original_data_checksum:
                                    self.logger.warning(f"Checksum fails for data iteration {step}. Check if the dataloader is deterministic!")
                                    compressed_data = None

            # run codec compress
            if compressed_data is None:
                time_start = time.time()
                compressed_data = self.codec.compress(data)
                time_compress = time.time() - time_start

            # compressed_string = pickle.dumps(compressed_data)
            compressed_length = len(compressed_data)
            # compressed_bits = compressed_length * 8

            # TODO: the data format should be defined!
            original_string = pickle.dumps(data)
            original_length = len(original_string)
            # original_bits = original_length * 8

            compression_ratio = compressed_length / original_length

            metric_logger.update(
                compression_ratio=compression_ratio,
                time_compress=time_compress*1000,
                speed_compress=(original_length/time_compress/1024/1024),
            )

            if not self.skip_decompress:
                time_start = time.time()
                decompressed_data = self.codec.decompress(compressed_data)
                time_decompress = time.time() - time_start

                metric_logger.update(
                    time_decompress=time_decompress*1000,
                    speed_decompress=(original_length/time_decompress/1024/1024),
                )
                # TODO: check decompress correctness

            # cache compressed data
            if self.cache_dir:
                if self.cache_compressed_data:
                    cache_file = os.path.join(self.cache_dir, "{}.bin".format(step))
                    if not os.path.exists(cache_file):
                        with open(cache_file, 'wb') as f:
                            if self.cache_checksum:
                                original_data_checksum = hashlib.md5(data).digest()
                                cached_data = (compressed_data, original_data_checksum)
                            else:
                                cached_data = compressed_data
                            pickle.dump(cached_data, f)

        return metric_logger.get_global_average()


class BasicLosslessCompressionBenchmark(BaseBenchmark):
    def __init__(self, codec: CodecInterface, dataloader: DataLoaderInterface, *args,
                 metric_logger=None,  # TODO: an interface for metric_logger
                 add_intermediate_to_metric=True,
                 # training
                 need_training=False,
                 training_dataloader=None,
                 trainer=None,
                 training_config=dict(),
                 load_checkpoint=True,
                 save_checkpoint=True,
                 checkpoint_file="params.pkl",
                #  max_training_samples=-1,
                 # testing
                 extra_testing_dataloaders=[],
                 cache_compressed_data=False,
                 cache_checksum=True,
                 cache_subdir="cache",
                 skip_decompress=False,
                 distortion_metric : Optional[BaseMetric] = None,
                 num_repeats=1,
                 num_testing_workers=0,
                 skip_trainer_testing=False,
                 force_basic_testing=False,
                 force_testing_device=None,
                 **kwargs):

        # output_dir for saving results
        self.extra_testing_dataloaders = extra_testing_dataloaders
        self.cache_compressed_data = cache_compressed_data
        self.cache_checksum = cache_checksum
        self.cache_subdir = cache_subdir
        # moved to setup_engine
        # self.cache_dir = os.path.join(self.output_dir, cache_subdir)
        # if self.cache_compressed_data and not os.path.exists(self.cache_dir):
        #     os.makedirs(self.cache_dir)

        # benchmark flow
        self.skip_decompress = skip_decompress
        self.distortion_metric = distortion_metric

        self.num_repeats = num_repeats
        self.num_testing_workers = num_testing_workers
        if num_testing_workers == -1:
            self.num_testing_workers = multiprocessing.cpu_count()
        self.skip_trainer_testing = skip_trainer_testing
        self.force_basic_testing = force_basic_testing
        self.force_testing_device = force_testing_device

        # if self.num_testing_workers > 0:
        #     self.testing_pool = multiprocessing.Pool(self.num_testing_workers)
        # else:
        #     self.testing_pool = None

        # metric logger
        self.metric_logger = MetricLogger() if metric_logger is None else metric_logger
        self.metric_logger.add_meter("compression_ratio", SmoothedValue(fmt="{global_avg:.4f}"))
        self.metric_logger.add_meter("time_compress", SmoothedValue(fmt="{global_avg:.2f} ms"))
        self.metric_logger.add_meter("speed_compress", SmoothedValue(fmt="{global_avg:.2f} MB/s"))
        self.metric_logger.add_meter("time_decompress", SmoothedValue(fmt="{global_avg:.2f} ms"))
        self.metric_logger.add_meter("speed_decompress", SmoothedValue(fmt="{global_avg:.2f} MB/s"))

        # an intermediate logger for finding bottlenecks
        self.itmd_logger = MetricLogger()
        self.itmd_logger.add_meter("time_dataloader", SmoothedValue(fmt="{median:.2f} ({global_avg:.2f}) ms"))
        self.itmd_logger.add_meter("time_iter", SmoothedValue(fmt="{median:.2f} ({global_avg:.2f}) ms"))
        
        self.add_intermediate_to_metric = add_intermediate_to_metric
        
        self.need_training = need_training
        self.training_dataloader = dataloader if training_dataloader is None else training_dataloader
        
        # setup trainer
        if trainer is None:
            trainer = BasicTrainer(self.training_dataloader, **training_config)
        # elif isinstance(trainer, BaseEngine):
        #     trainer.setup_engine_from_copy(self)
        # else:
        #     raise ValueError("Invalid trainer!")
        self.trainer = trainer
        self.training_config = training_config
        self.load_checkpoint = load_checkpoint
        self.save_checkpoint = save_checkpoint
        self.checkpoint_file = checkpoint_file
        # self.max_training_samples = max_training_samples

        # initialize benchmark engine
        super().__init__(codec, dataloader, *args, **kwargs)


    def setup_engine(self, *args, output_dir=None, **kwargs):
        super().setup_engine(*args, output_dir=output_dir, **kwargs)
        # setup cache output
        if self.output_dir is not None:
            self.cache_dir = os.path.join(self.output_dir, self.cache_subdir)
            if self.cache_compressed_data and not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
        else:
            self.logger.warning("Cache dir not properly setup!")
        # setup trainer output
        if isinstance(self.trainer, BaseEngine):
            self.trainer.setup_engine(*args, output_dir=output_dir, **kwargs)


    def set_codec(self, codec) -> None:
        super().set_codec(codec)
        # inject profiler (seems it is not needed as all codecs are BaseModule which has it own profiler)
        # if isinstance(self.codec, BaseCodec):
        #     self.itmd_logger.reset()
        #     self.codec.profiler = self.itmd_logger

        # training
        if not isinstance(self.codec, TrainableModuleInterface) and self.need_training:
            self.logger.warning("Codec is not trainable! Skip training!")
            self.need_training = False

    def _estimate_byte_length(self, data):
        import torch
        # a lazy solution for evaluating total bytes...
        # pickle.dumps add some useless information such as class definitions!
        # return len(pickle.dumps(data))
        if isinstance(data, bytes):
            return len(data) # TODO: length of string?
        elif isinstance(data, str):
            return len(data.encode('utf-8')) # TODO: length of string?
        elif isinstance(data, torch.Tensor):
            return int(np.prod(data.shape)) * data.element_size()
        elif isinstance(data, np.ndarray):
            return int(data.size * data.itemsize)
        else:
            raise ValueError("Bitstream of data {} in type {} cannot be estimated!".format(data, type(data)))

    def _run_step(self, step: int, data: Any):
        # search cache
        compressed_data = None
        if self.output_dir:
            if self.cache_compressed_data:
                # NOTE: maybe using hash values and filenames? dataloader may be shuffled!
                cache_file = os.path.join(self.cache_dir, "{}.bin".format(step))
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        compressed_data = pickle.load(f)
                        if self.cache_checksum:
                            compressed_data, checksum = compressed_data
                            original_data_checksum = hashlib.md5(data).digest()
                            # checksum fail! rerun codec!
                            if checksum != original_data_checksum:
                                self.logger.warning(f"Checksum fails for data iteration {step}. Check if the dataloader is deterministic!")
                                compressed_data = None

        # run codec compress
        if compressed_data is None:
            time_start = time.time()
            compressed_data = self.codec.compress(data)
            time_compress = time.time() - time_start

        try:
            compressed_length = self._estimate_byte_length(compressed_data) 
        except ValueError:
            self.logger.warn("Cannot estimate compressed data length! Using pickle as an alt way.")
            compressed_string = pickle.dumps(compressed_data)
            compressed_length = len(compressed_string)
        # compressed_bits = compressed_length * 8

        # TODO: the data format should be defined!
        try:
            original_length = self._estimate_byte_length(data) 
        except ValueError:
            self.logger.warn("Cannot estimate original data length! Using pickle as an alt way.")
            original_string = pickle.dumps(data)
            original_length = len(original_string)
            # original_bits = original_length * 8

        compression_ratio = compressed_length / original_length

        self.metric_logger.update(
            compression_ratio=compression_ratio,
            compressed_length=compressed_length,
            original_length=original_length,
            time_compress=time_compress*1000,
            speed_compress=(original_length/time_compress/1024/1024),
        )

        if not self.skip_decompress:
            time_start = time.time()
            decompressed_data = self.codec.decompress(compressed_data)
            time_decompress = time.time() - time_start

            # NOTE: use original_length or compressed_length when calculating speed_decompress?
            self.metric_logger.update(
                time_decompress=time_decompress*1000,
                speed_decompress=(original_length/time_decompress/1024/1024),
            )
            # check decompress correctness for lossless
            # assert(data == decompressed_data)

            # distortion
            if self.distortion_metric is not None:
                # if isinstance(self.distortion_metric, dict):
                #     for name, method in self.distortion_metric.items():
                #         self.metric_logger[f'distortion_{name}'] = method(data, decompressed_data)
                # else:
                metric_dict = self.distortion_metric(data, decompressed_data)
                self.metric_logger.update(**metric_dict)


        # cache compressed data
        if self.output_dir:
            if self.cache_compressed_data:
                cache_file = os.path.join(self.cache_dir, "{}.bin".format(step))
                if not os.path.exists(cache_file):
                    with open(cache_file, 'wb') as f:
                        if self.cache_checksum:
                            original_data_checksum = hashlib.md5(data).digest()
                            cached_data = (compressed_data, original_data_checksum)
                        else:
                            cached_data = compressed_data
                        pickle.dump(cached_data, f)            
        
        # reset cache to free memory?
        if isinstance(self.codec, NNTrainableModule):
            self.codec.reset_all_cache()

    def run_training(self, *args, **kwargs):
        # training step
        if self.trainer is not None:
            training_config = dict(
                load_checkpoint=self.load_checkpoint,
                save_checkpoint=self.save_checkpoint,
                checkpoint_file=self.checkpoint_file
            )
            training_config.update(**self.training_config)
            self.trainer.train_module(self.codec, *args,
                **training_config, **kwargs
            )
            # checkpoint_path = os.path.join(self.output_dir, self.checkpoint_file)
            # # load checkpoint
            # if self.load_checkpoint and os.path.exists(checkpoint_path):
            #     self.logger.info("Loading checkpoint from {} ...".format(checkpoint_path))
            #     with open(checkpoint_path, 'rb') as f:
            #         params = pickle.load(f)
            #         self.codec.load_parameters(params)
            # else:
            #     self.logger.info("Start Training!")
            #     if self.trainer == "fulldata":
            #         # TODO: use a special training dataloader
            #         training_samples = list(self.dataloader)
            #         if self.max_training_samples:
            #             random.shuffle(training_samples)
            #             training_samples = training_samples[:self.max_training_samples]
            #         self.codec.train_full(training_samples)
            #     else:
            #         raise ValueError("Trainer {} not supported!".format(self.trainer))

            #     # save checkpoint
            #     if self.save_checkpoint:
            #         self.logger.info("Saving checkpoint to {} ...".format(checkpoint_path))
            #         params = self.codec.get_parameters()
            #         with open(checkpoint_path, 'wb') as f:
            #             pickle.dump(params, f)

    def run_testing(self, dataloader : DataLoaderInterface, *args, skip_trainer_testing=False, **kwargs):
        if self.codec is None:
            raise ValueError("No codec to benchmark!")

        self.logger.info("Starting benchmark testing!")
        metrics = dict()
        try:
            # raise NotImplementedError()
            # custom trainer testing
            # TODO: metric logger?
            if self.trainer is not None:
                metrics_trainer = dict()
                if not skip_trainer_testing:
                    metrics_trainer = self.trainer.test_module(self.codec,
                        **self.training_config # TODO: may use independent config for testing!
                    )
                else:
                    # just load checkpoint
                    self.trainer.load_checkpoint(self.codec, checkpoint_file=self.checkpoint_file)
                if skip_trainer_testing or self.force_basic_testing:
                    if isinstance(metrics_trainer, dict):
                        metrics.update(**metrics_trainer)
                    else:
                        self.logger.warn("No metric collected from trainer testing procedure!")
                else:
                    return metrics_trainer
                
        except NotImplementedError:
            self.logger.info("Using default benchmark testing!")


        for _ in range(self.num_repeats):

            # Use default benchmark testing
            worker = _BenchmarkTestingWorker(self.codec, dataloader,
                # metric_logger=self.metric_logger,
                cache_dir=self.cache_dir,
                cache_checksum=self.cache_checksum,
                cache_compressed_data=self.cache_compressed_data,
                skip_decompress=self.skip_decompress,
            )
            time_start = time.time()
            # data_cache = []

            can_pickle = True
            try:
                pickle.dumps(worker)
            except:
                self.logger.warn("Cannot pickle worker! Multiprocessing disabled!")
                can_pickle = False

            # disable multiprocessing if we need to test on non-cpu device
            if self.num_testing_workers > 0 and can_pickle and self.force_testing_device is None:
                with multiprocessing.Pool(self.num_testing_workers) as testing_pool:
                    # dataloader_segment = len(dataloader) // self.num_testing_workers
                    # dataloader_split = [enumerate(dataloader[(i*dataloader_segment):((i+1)*dataloader_segment)], start=i*dataloader_segment) for i in range(self.num_testing_workers)]
                    # metric_data = self.testing_pool.starmap(worker, dataloader_split)
                    # for metric_dict in metric_data:
                    #     self.metric_logger.update(**metric_dict)
                    # data_cache = []
                    # async_results = []
                    # for i, data in enumerate(dataloader):
                    #     data_cache.append((i, data))
                    #     if len(data_cache) > 100:
                    #         async_results.append(self.testing_pool.starmap_async(worker, [data_cache]))
                    #         data_cache = []
                    num_segments = self.num_testing_workers # * 10
                    segment_length = len(dataloader) // num_segments
                    dataloader_split = [range(i*segment_length, min((i+1)*segment_length, len(dataloader))) for i in range(num_segments)]
                    # for split in tqdm(dataloader_split):
                    #     result = worker(split)
                    results = testing_pool.map(worker, dataloader_split)
                    for result in results:
                        self.metric_logger.update(**result)
                    # results = [self.testing_pool.apply_async((split, )) for split in dataloader_split]
                    # for result in tqdm(results):
                    #     try:
                    #         metric_dict = result.get()
                    #         self.metric_logger.update(**metric_dict)
                    #     except:
                    #         traceback.print_exc()
            else:
                # TODO: when using distributed training, the distributed backend could not end propoerly
                # which causes testing to run on multiple processes.
                # this is just a temporary fix!
                # NOTE: it seems this causes deadlock...
                # if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0: break
                
                # for trainable codecs
                if isinstance(self.codec, TrainableModuleInterface):
                    if isinstance(self.codec, NNTrainableModule):
                        self.codec.eval()
                        if self.force_testing_device is not None:
                            try:
                                self.codec.to(device=self.force_testing_device)
                            except:
                                self.logger.warn(f"Device {self.force_testing_device} not available on this machine! Testing on default CPU!")
                    self.codec.update_state()
                
                for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                    # if not i in [38, 47, 82, 86, 92, 97]: continue
                    # if i != 92: continue
                    # print(i)
                    time_dataloader = time.time() - time_start
                    self.itmd_logger.update(time_dataloader=time_dataloader*1000)

                    with self.itmd_logger.start_time_profile("time_iter"):
                        # if self.testing_pool is not None:
                        #     data_cache.append((i, data))
                        #     if len(data_cache) == self.num_testing_workers or i == len(dataloader)-1:
                        #         metric_data = self.testing_pool.starmap(worker, data_cache)
                        #         for metric_dict in metric_data:
                        #             self.metric_logger.update(**metric_dict)
                        #         data_cache = []
                        # else:
                        try:
                            compressed_data = self._run_step(i, data)
                        except Exception as e:
                            self.logger.error("Basic Testing found an error! Aborting...")
                            self.logger.error(traceback.format_exc())
                            break
                            # compressed_data = worker(i, data)

                    if i % 1000 == 0:
                        self.logger.info(self.itmd_logger)
                    
                    time_start = time.time()

            self.metric_logger.synchronize_between_processes()

            # update metrics
            metrics = self.metric_logger.get_global_average()

            # if hasattr(self.codec, "profiler"):
            #     self.logger.info("Running Intermediate Logger: {}".format(self.codec.profiler))

            if self.add_intermediate_to_metric:
                if isinstance(self.codec, BaseCodec):
                    intermediate_metrics = self.codec.collect_profiler_results(recursive=True)
                    self.logger.info("Running Intermediate Logger: {}".format(
                            json.dumps(
                                intermediate_metrics, 
                                indent=4
                            )
                        )
                    )
                    metrics.update(**intermediate_metrics)
        
        return metrics


    def run_benchmark(self, *args,
        run_training=True,
        run_testing=True,
        ignore_exist_metrics=False,
        **kwargs):
        if self.codec is None:
            raise ValueError("No codec to benchmark!")

        # check if the benchmark has been run by checking metric file
        if not ignore_exist_metrics and os.path.exists(self.metric_raw_file):
            self.logger.warning("Metric file {} already exists! Skipping benchmark...".format(self.metric_raw_file))
            self.logger.warning("Specify ignore_exist_metrics=True if you want to restart the benchmark.")
            # read metric file as output
            metric_data = None
            with open(self.metric_raw_file, 'rb') as f:
                # read the first line
                # metric_data = next(csv.DictReader(f))
                metric_data = pickle.load(f)
            
            return metric_data
        
        self.metric_logger.reset()

        if run_training:
            self.run_training(*args, **kwargs)

        if run_testing:
            metric_dict = self.run_testing(self.dataloader, *args, skip_trainer_testing=self.skip_trainer_testing, **kwargs)
            if len(self.extra_testing_dataloaders) > 0:
                for extra_id, dataloader in enumerate(self.extra_testing_dataloaders):
                    extra_metric_dict = self.run_testing(dataloader, *args, skip_trainer_testing=True, **kwargs)
                    metric_dict.update(**{(f"extra{extra_id}_{k}"):v for k, v in extra_metric_dict.items()})
            self.save_metrics(metric_dict)
            return metric_dict

    def collect_metrics(self, *args, **kwargs):
        # TODO: this may be invalid when using custom test_module function!
        return self.metric_logger.get_global_average()


class GroupedLosslessCompressionBenchmark(BasicLosslessCompressionBenchmark):
    def __init__(self, codec_group: Union[CodecInterface, List[CodecInterface]],
                 dataloader: DataLoaderInterface,
                 *args,
                 **kwargs):
        if isinstance(codec_group, CodecInterface):
            codec_group = [codec_group]
        self.codec_group = codec_group
        super().__init__(codec_group[0], dataloader, *args, **kwargs)

        self.cached_metrics = []

    def run_benchmark(self, *args, **kwargs):
        self.cached_metrics = []
        for idx, codec in enumerate(self.codec_group):
            if self.trainer is not None:
                codec_output_dir = os.path.join(self.output_dir, "codec{}".format(idx))
                self.trainer.setup_engine(output_dir=codec_output_dir)
            self.codec = codec
            metric_dict = super().run_benchmark(*args, **kwargs)
            self.cached_metrics.append(metric_dict)

        return self.collect_metrics()

    def collect_metrics(self, *args, **kwargs):
        # TODO: draw a plot?
        return self.cached_metrics