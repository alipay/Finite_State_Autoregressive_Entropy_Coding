import logging
import threading
import traceback
from typing import Iterable
from cbench.utils.logger import setup_logger

from configs.class_builder import NamedParamBase, ClassBuilder, ClassBuilderList
from configs.import_utils import import_config_from_file
from configs.env import DEFAULT_EXPERIMENT_PATH, DEFAULT_OSS_EXPERIMENT_PATH
from configs.oss_utils import OSSUtils

import os
import time
import csv
import pickle
import datetime
import hashlib
import multiprocessing
import platform
import asyncio
from contextlib import suppress

tensorboard_enabled = False
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    tensorboard_enabled = True
except:
    pass

class FileSyncObject(object):
    def __init__(self, local_dir, *args, oss_dir=None, interval=600, logger=None, **kwargs) -> None:
        self.local_dir = local_dir
        self.oss_dir = oss_dir if oss_dir is not None else local_dir
        self.interval = interval
        self.logger = logger

        self.exit_event = threading.Event()

    def upload(self):
        oss = OSSUtils(logger=self.logger)
        oss.upload_directory(self.oss_dir, self.local_dir, force_overwrite_dir=True)

    def download(self):
        oss = OSSUtils(logger=self.logger)
        oss.download_directory(self.oss_dir, self.local_dir)

    def request_exit(self):
        self.exit_event.set()

    # multiprocessing enter point
    def __call__(self) -> None:
        while True:
            if self.exit_event.is_set(): break
            self.upload()
            # allow exit event when sleeping
            for i in range(int(self.interval)):
                if self.exit_event.is_set(): 
                    return
                time.sleep(1)
            # await asyncio.sleep(self.interval)


class Periodic:
    def __init__(self, func, time):
        self.func = func
        self.time = time
        self.is_started = False
        self._task = None

    async def start(self):
        if not self.is_started:
            self.is_started = True
            # Start task to call func periodically:
            self._task = asyncio.ensure_future(self._run())

    async def stop(self):
        if self.is_started:
            self.is_started = False
            # Stop task and await it stopped:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self):
        while True:
            await asyncio.sleep(self.time)
            self.func()


def run_config(config: NamedParamBase, args, exp_name=None):

    logger = setup_logger("run_benchmark")

    # batch config?
    if isinstance(config.param, ClassBuilderList):
        for config_single in config.param:
            run_config(config_single, args)
        return

    # process config
    config_name = config.name
    # get experiment name
    exp_name_full = args.exp_name if exp_name is None else exp_name
    if not exp_name_full:
        # time_string = datetime.datetime.now()
        # config_hash = hash(config)
        exp_name_full = config_name

    logger.info("Experiment Name:")
    logger.info(exp_name_full)

    # avoid filename too long 
    exp_name_length_limit = 64
    if len(exp_name_full) > exp_name_length_limit:
        hash_length = 16
        config_hash = hashlib.sha256(config.name.encode()).hexdigest()[:hash_length]
        hash_prefix = f"{config_hash}"
        exp_name = f"{config_hash}({exp_name_full[:(exp_name_length_limit-len(config_hash))]}..."
    else:
        exp_name = exp_name_full

    seed = args.seed
    # repeated benchmark. support distributed running with different output dirs
    if args.repeat_idx > 0:
        exp_name = os.path.join(exp_name, f"repeat_{args.repeat_idx}")
        seed += args.repeat_idx

    output_dir = os.path.join(args.output_dir, exp_name)

    # update output_dir
    # output_dir = config.get_parameter("output_dir")
    # if output_dir is None:
    # config = config.update_args(
    #     output_dir=output_dir,
    #     # load_checkpoint=(not args.force_restart),
    # )
    config_dir = os.path.join(output_dir, "config.pkl")
    config_name_dir = os.path.join(output_dir, "config_name.txt")
    exp_name_dir = os.path.join(output_dir, "exp_name.txt")

    logger.info("Output dir:")
    logger.info(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # maybe we should check if the config has changed?
        # not required now as the config name already does this
        if not args.force_override and os.path.exists(config_name_dir):
            with open(config_name_dir, 'r') as f:
                config_name_original = f.read()
                if config_name_original != config_name:
                    raise ValueError(f"Output dir {output_dir} has already been used by another config! Specify --force-override if you want to override the old config!")
        # pass
        # if not args.force_override and not args.force_restart:
        #     print("Output dir already exists! Exiting...")
        #     print("To force running this experiment, specify -f or -fr.")
        #     return

    # save config to file
    # TODO: if there's an existing config file, check if the config is equal!
    with open(config_dir, 'wb') as f:
        pickle.dump(config, f)
        print(f"Config saved to {config_dir}")
    
    # save config and exp name
    with open(config_name_dir, 'w') as f:
        f.write(config_name)
        print(f"Config name saved to {config_name_dir}")
    with open(exp_name_dir, 'w') as f:
        f.write(exp_name_full)
        print(f"Experiment name saved to {exp_name_dir}")

    file_sync_task = None
    if args.sync and multiprocessing.current_process().name == "MainProcess":
        oss_dir = os.path.join(args.sync_dir, exp_name)
        # open a new process for uploading checkpoints
        file_sync_obj = FileSyncObject(output_dir, 
            oss_dir=oss_dir, 
            interval=args.sync_interval,
            logger=setup_logger("OSSUtils", log_level='WARNING', log_level_file='INFO', outdir=output_dir, label='oss_utils')
        )
        file_sync_obj.download()
        if args.sync_interval > 0:
            file_sync_task = threading.Thread(
                target=file_sync_obj, name="file_sync_task"
            )
            file_sync_task.start()

    if file_sync_task is not None:
        try:
            run_benchmark(config, logger, output_dir, exp_name, args, seed=seed)
        except Exception as e:
            traceback.print_exc()
        finally:
            # file_sync_task.terminate()
            file_sync_obj.request_exit()
            # sometimes request_exit fail to work, just join with timeout
            file_sync_task.join(timeout=600)
            logger.info("file_sync_task successfully joined!")
            # file_sync_task.join(timeout=0)
            # file_sync_task.cancel()
            # await file_sync_task
            # await file_sync_task.stop()
    else:
        run_benchmark(config, logger, output_dir, exp_name, args, seed=seed)

    if args.sync and multiprocessing.current_process().name == "MainProcess":
        # final upload everything to oss if available
        file_sync_obj.upload()


def run_config_dir(path, args):
    output_dir = path
    logger = setup_logger("run_benchmark")
    logger.info("Output dir:")
    logger.info(output_dir)

    file_sync_task = None
    if args.sync and multiprocessing.current_process().name == "MainProcess":
        oss_dir = path
        # open a new process for uploading checkpoints
        file_sync_obj = FileSyncObject(output_dir, 
            oss_dir=oss_dir, 
            interval=args.sync_interval,
            logger=setup_logger("OSSUtils", log_level='WARNING', log_level_file='INFO', outdir=output_dir, label='oss_utils')
        )
        file_sync_obj.download()
        if args.sync_interval > 0:
            file_sync_task = threading.Thread(
                target=file_sync_obj, name="file_sync_task"
            )
            file_sync_task.start()

    with open(os.path.join(output_dir, "config.pkl"), 'rb') as f:
        config = pickle.load(f)
    
    # read exp name
    with open(os.path.join(output_dir, "exp_name.txt"), 'r') as f:
        exp_name = f.read()
    
    if file_sync_task is not None:
        try:
            run_benchmark(config, logger, output_dir, exp_name, args)
        except Exception as e:
            traceback.print_exc()
        finally:
            # file_sync_task.terminate()
            file_sync_obj.request_exit()
            # sometimes request_exit fail to work, just join with timeout
            file_sync_task.join(timeout=600)
            logger.info("file_sync_task successfully joined!")
            # file_sync_task.join(timeout=0)
            # file_sync_task.cancel()
            # await file_sync_task
            # await file_sync_task.stop()
    else:
        run_benchmark(config, logger, output_dir, exp_name, args)

    if args.sync and multiprocessing.current_process().name == "MainProcess":
        # final upload everything to oss if available
        file_sync_obj.upload()


def run_benchmark(config, logger, output_dir, exp_name, args, seed=None):

    # skip if metric file exists
    metric_file = os.path.join(output_dir, "metrics.csv")
    if not args.ignore_exist_metrics and os.path.exists(metric_file):
        logger.warning("Metric file {} already exists! Skipping benchmark...".format(metric_file))
        logger.warning("Specify -im if you want to restart the benchmark.")
        return
        

    # setup file sync process for cloud based benchmarking
    # possible fix for multiprocess
    # TODO: use python asyncio instead of multiprocessing!
    # file_sync_task = None
    # if args.sync and multiprocessing.current_process().name == "MainProcess":
    #     oss_dir = os.path.join(args.sync_dir, exp_name)
    #     # first download checkpoint if available
    #     # oss = OSSUtils(
    #     #     logger=setup_logger("OSSUtils", outdir=output_dir, label='oss_utils')
    #     # )
    #     # oss.download_directory(oss_dir, output_dir)
    #     # open a new process for uploading checkpoints
    #     file_sync_obj = FileSyncObject(output_dir, 
    #         oss_dir=oss_dir, 
    #         interval=args.sync_interval,
    #         logger=setup_logger("OSSUtils", log_level='WARNING', log_level_file='INFO', outdir=output_dir, label='oss_utils')
    #     )
    #     file_sync_obj.download()
    #     if args.sync_interval > 0:
    #         # file_sync_task = asyncio.create_task(file_sync_obj())
    #         # file_sync_task = Periodic(file_sync_obj.upload, args.sync_interval)
    #         # await file_sync_task.start()
    #         # file_sync_task = multiprocessing.Process(
    #         #     target=file_sync_obj
    #         # )
    #         file_sync_task = threading.Thread(
    #             target=file_sync_obj, name="file_sync_task"
    #         )
    #         file_sync_task.start()

    # build class from config and start benchmarking
    benchmark_logger = setup_logger("benchmark_logger", outdir=output_dir, label="benchmark")

    benchmark = config.build_class(
        output_dir=output_dir,
        logger=benchmark_logger,
        load_checkpoint=(not args.restart_training), # TODO: this is not logical!
    )
    # print(benchmark.output_dir)

    run_training = not args.test_only
    run_testing = not args.train_only
    if seed is None:
        seed = args.seed

    # if file_sync_task is not None:
    #     try:
    #         metric_logger = benchmark.run_benchmark(
    #             run_training=run_training,
    #             run_testing=run_testing,
    #             force_restart=args.force_restart,
    #             ignore_exist_metrics=args.ignore_exist_metrics,
    #             codecs_ignore_exist_metrics=args.codecs_ignore_exist_metrics,
    #             # restart_training=args.restart_training,
    #         )
    #         logger.info("Final results:")
    #         logger.info(metric_logger)
    #     except Exception as e:
    #         traceback.print_exc()
    #     finally:
    #         # file_sync_task.terminate()
    #         file_sync_obj.request_exit()
    #         # sometimes request_exit fail to work, just join with timeout
    #         file_sync_task.join(timeout=600)
    #         logger.info("file_sync_task successfully joined!")
    #         # file_sync_task.join(timeout=0)
    #         # file_sync_task.cancel()
    #         # await file_sync_task
    #         # await file_sync_task.stop()
    # else:
    # easier for debugger to break on exception
    metric_logger = benchmark.run_benchmark(
        run_training=run_training,
        run_testing=run_testing,
        force_restart=args.force_restart,
        ignore_exist_metrics=args.ignore_exist_metrics,
        codecs_ignore_exist_metrics=args.codecs_ignore_exist_metrics,
        initial_seed=seed,
        # restart_training=args.restart_training,
    )

    logger.info("Final results:")
    logger.info(metric_logger)

    # save metric file
    metrics = benchmark.collect_metrics()
    if metrics is not None:
        try:
            if isinstance(metrics, (list, tuple)):
                fieldnames = list(metrics[0].keys())
                metric_data = metrics
            else:
                fieldnames = list(metrics.keys())
                metric_data = [metrics]
            if not os.path.exists(metric_file):
                with open(metric_file, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(metric_data)
                    logger.info(f"Metrics saved to {metric_file}")
        except Exception as e:
            # traceback.print_exc()
            logger.error("Metrics are invalid!")

    # if args.sync and multiprocessing.current_process().name == "MainProcess":
    #     # final upload everything to oss if available
    #     file_sync_obj.upload()
    #     # oss = OSSUtils()
    #     # oss.upload_directory(output_dir, output_dir)


def main(args):
    for cfg_path in args.config_files_or_dir:
        if os.path.isfile(cfg_path):
            config = import_config_from_file(cfg_path, set_module_name=False)
            run_config(config, args)
        else:
            run_config_dir(cfg_path, args)
        # asyncio.run(run_config(config, args))
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(run_config(config, args))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_files_or_dir", nargs='+', default=['configs/default.py'])
    parser.add_argument("--output-dir", '-o', type=str, default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument("--exp-name", '-n', type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-idx", '-ri', type=int, default=0)
    parser.add_argument("--force-override", '-f', action="store_true")
    parser.add_argument("--force-restart", '-fr', action="store_true")
    parser.add_argument("--restart-training", '-rt', action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--ignore-exist-metrics", '-im', action="store_true")
    parser.add_argument("--codecs-ignore-exist-metrics", '-cim', action="store_true")
    parser.add_argument("--sync", '-s', action="store_true")
    parser.add_argument("--sync-interval", '-si', type=int, default=0)
    parser.add_argument("--sync-dir", '-sd', type=str, default=DEFAULT_OSS_EXPERIMENT_PATH)

    args = parser.parse_args()

    # fix for macos multiprocessing
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')
    main(args)
