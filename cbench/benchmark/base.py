import abc
from collections import OrderedDict
from ctypes import Union
import logging
import os
import pickle
import sys
import csv
from typing import Optional, Union, List, Dict

from cbench.codecs.base import CodecInterface
from cbench.data.base import DataLoaderInterface
from cbench.utils.engine import BaseEngine

tensorboard_enabled = False
# NOTE: tmp disabled
# try:
#     from torch.utils.tensorboard.writer import SummaryWriter
#     tensorboard_enabled = True
# except:
#     pass

class BaseBenchmark(BaseEngine):
    def __init__(self, codec: CodecInterface, dataloader: DataLoaderInterface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codec = codec
        self.dataloader = dataloader
        
    @abc.abstractmethod
    def run_benchmark(self, *args, **kwargs) -> Union[Dict, List[Dict]]:
        pass

    @abc.abstractmethod
    def collect_metrics(self, *args, **kwargs) -> Union[Dict, List[Dict], None]:
        pass

    def set_codec(self, codec) -> None:
        # may perform some extra actions in children
        self.codec = codec

    @property
    def metric_file(self) -> str:
        return os.path.join(self.output_dir, "metrics.csv")

    @property
    def metric_raw_file(self) -> str:
        return os.path.join(self.output_dir, "metrics.pkl")

    @property
    def metric_log_dir(self) -> str:
        return os.path.join(self.output_dir, "metric_logs")


    def save_metrics(self, metric_data=None, hparams=None, names=None) -> None:
        metric_file = self.metric_file
        if metric_data is None:
            metric_data = self.collect_metrics()
        
        if not metric_data is None:
            fieldnames = OrderedDict()
            fieldnames_metrics = OrderedDict()
            fieldnames_hparams = OrderedDict()
            metrics_to_write = []

            with open(self.metric_raw_file, 'wb') as f:
                pickle.dump(metric_data, f)

            if tensorboard_enabled:
                tb_writer = SummaryWriter(self.metric_log_dir)

            def _append_metric_and_hparam(metric, hparam=None, name=None):
                write_dict = OrderedDict()
                if isinstance(metric, dict):
                    # write exp names in the first column for a clear view
                    if name is not None:
                        fieldnames['name'] = None
                        write_dict['name'] = name
                    for k in metric:
                        fieldnames_metrics[k] = None
                    write_dict.update(**metric)
                    if isinstance(hparam, dict):
                        for k in hparam:
                            fieldnames_hparams[k] = None
                        write_dict.update(**hparam)
                        # filter hparams to avoid tb error
                        hparam_filtered = {k:v for k, v in hparam.items() if isinstance(v, (int, float, str, bool))}
                        if tensorboard_enabled:
                            tb_writer.add_hparams(hparam_filtered, dict(**metric), run_name=name)
                metrics_to_write.append(write_dict)

            if isinstance(metric_data, (list, tuple)):
                if hparams is None:
                    hparams = [None] * len(metric_data)
                if names is None:
                    names = [None] * len(metric_data)
                for metric_dict, hparam_dict, name in zip(metric_data, hparams, names):
                    _append_metric_and_hparam(metric_dict, hparam_dict, name)
            else:
                _append_metric_and_hparam(metric_data, hparams, names)

            fieldnames = list(fieldnames.keys()) + list(fieldnames_metrics.keys()) + list(fieldnames_hparams.keys())
            with open(metric_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics_to_write)
                self.logger.info(f"Metrics saved to {metric_file}")


