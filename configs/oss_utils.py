import base64
import logging
import time
import oss2
import os, shutil
from pathlib import Path
import tempfile

# TODO: support multiprocessing upload and download
import multiprocessing
import functools

from oss2.utils import Crc64, make_crc_adapter

from configs.env import DEFAULT_OSS_ENDPOINT, DEFAULT_OSS_BUCKET_NAME, DEFAULT_OSS_KEYID_BASE64, DEFAULT_OSS_KEYSEC_BASE64, DEFAULT_OSS_PERSONAL_ROOT
from configs.env import DEFAULT_CPU_CORES

# https://lscsoft.docs.ligo.org/bilby/_modules/bilby/core/utils/log.html#setup_logger
def setup_logger(name="default", outdir=None, label=None, log_level='INFO', log_level_file='DEBUG', print_version=False):
    """ Setup logging output: call at the start of the script to use

    Parameters
    ==========
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    log_level_file: str, optional
        similar to log_level, log level for logfile
    print_version: bool
        If true, print version information
    """

    if type(log_level) is str:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError('log_level {} not understood'.format(log_level))
    else:
        level = int(log_level)

    if type(log_level_file) is str:
        try:
            level_file = getattr(logging, log_level_file.upper())
        except AttributeError:
            raise ValueError('log_level_file {} not understood'.format(log_level_file))
    else:
        level_file = int(log_level_file)

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)

    if any([type(h) == logging.StreamHandler for h in logger.handlers]) is False:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
        if label is not None and outdir is not None:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            log_file = '{}/{}.log'.format(outdir, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

            file_handler.setLevel(level_file)
            logger.addHandler(file_handler)

    # for handler in logger.handlers:
    #     handler.setLevel(level)

    return logger

class OSSUtils(object):
    def __init__(self, 
        keyId_b64=DEFAULT_OSS_KEYID_BASE64, 
        keySec_b64=DEFAULT_OSS_KEYSEC_BASE64, 
        endpoint=DEFAULT_OSS_ENDPOINT, 
        bucket_name=DEFAULT_OSS_BUCKET_NAME,
        personal_root=DEFAULT_OSS_PERSONAL_ROOT,
        max_retry=-1,
        num_process=DEFAULT_CPU_CORES,
        logger=None,
    ) -> None:
        self.keyId_b64 = keyId_b64
        self.keyId = base64.b64decode(keyId_b64).decode('utf-8')
        self.keySec_b64 = keySec_b64
        self.keySec = base64.b64decode(keySec_b64).decode('utf-8')
        self.endpoint = endpoint
        self.bucket_name = bucket_name

        self.personal_root = personal_root
        self.max_retry = max_retry
        # NOTE: pool cannot be pickled
        self.num_process = num_process
        if num_process > 0:
            self.pool = multiprocessing.Pool(num_process)

        self.logger = setup_logger("OSSUtils") if logger is None else logger

        self._connect_bucket()

    def _clone_wo_pool(self):
        return self.__class__(
            keyId_b64=self.keyId_b64, 
            keySec_b64=self.keySec_b64, 
            endpoint=self.endpoint, 
            bucket_name=self.bucket_name,
            personal_root=self.personal_root,
            max_retry=self.max_retry,
            num_process=0, # no pool
        )

    def _connect_bucket(self):
        auth = oss2.Auth(self.keyId, self.keySec)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

    def _checksum_file(self, file_obj):
        adapter = make_crc_adapter(file_obj)
        adapter.read()
        return adapter.crc

    def _compare_checksum(self, oss_path, local_path):
        oss_file = os.path.join(self.personal_root, oss_path)
        try:
            oss_crc = self.bucket.get_object(oss_file).server_crc
        except:
            self.logger.warning(f"Cannot get object {oss_file} checksum! Assuming a different checksum!")
            return False
        with open(local_path, 'rb') as f:
            local_crc = self._checksum_file(f.read())
            return oss_crc == local_crc

    def _iter_local_dir(self, local_dir : str):
        """ Iterate relative paths of local files

        Args:
            local_dir (str)

        Yields:
            str: relative file paths to local_dir
        """        
        for dirpath, dirnames, filenames in os.walk(local_dir):
            for fname in filenames:
                local_path = os.path.join(dirpath, fname)
                filename = os.path.relpath(local_path, local_dir)
                yield filename

    def _iter_oss_dir(self, oss_dir : str):
        """ Iterate relative paths of oss files

        Args:
            oss_dir (str): _description_

        Yields:
            str: relative file paths to oss_dir
        """        
        oss_prefix = os.path.join(self.personal_root, oss_dir) 
        if not oss_prefix.endswith("/"): oss_prefix += "/"
        for obj in oss2.ObjectIterator(self.bucket, prefix=oss_prefix):
            oss_path = os.path.relpath(obj.key, self.personal_root)
            filename = os.path.relpath(oss_path, oss_dir)
            yield filename

    def _upload_to_dir_by_relpath(self, oss_dir, local_dir, fpath):
        local_path = os.path.join(local_dir, fpath)
        oss_path = os.path.join(oss_dir, fpath)
        self.upload(oss_path, local_path)

    def _download_to_dir_by_relpath(self, oss_dir, local_dir, fpath):
        local_path = os.path.join(local_dir, fpath)
        oss_path = os.path.join(oss_dir, fpath)
        self.download(oss_path, local_path)

    def _delete_oss_by_relpath(self, oss_dir, fpath):
        oss_path = os.path.join(self.personal_root, oss_dir, fpath)
        self.bucket.delete_object(oss_path)

    def _delete_local_by_relpath(self, local_dir, fpath):
        local_path = os.path.join(local_dir, fpath)
        os.remove(local_path)

    def exists(self, oss_path) -> bool:
        oss_file = os.path.join(self.personal_root, oss_path)
        return self.bucket.object_exists(oss_file)

    def delete(self, oss_path):
        oss_file = os.path.join(self.personal_root, oss_path)
        self.bucket.delete_object(oss_file)

    def download(self, oss_path, local_path, *args, checksum=True, allow_overwrite=True, **kwargs):
        oss_file = os.path.join(self.personal_root, oss_path)
        retry_cnt = 0
        while retry_cnt < self.max_retry or self.max_retry < 0:
            try:
                if not self.bucket.object_exists(oss_file):
                    self.logger.warning("oss file {} not exist!".format(oss_path))
                    return
                if os.path.exists(local_path):
                    if not allow_overwrite:
                        self.logger.info("Local file exists! Skip downloading oss file {} to {}".format(oss_path, local_path))
                        return
                    elif checksum and self._compare_checksum(oss_path, local_path):
                        self.logger.info("Same checksum! Skip downloading oss file {} to {}".format(oss_path, local_path))
                        return
                # download
                self.logger.info("Download oss file {} to {}".format(oss_path, local_path))
                local_dir = os.path.dirname(os.path.abspath(local_path))
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir, exist_ok=True)
                self.bucket.get_object_to_file(oss_file, local_path, *args, **kwargs)
                return
            except oss2.exceptions.RequestError:
                # retry uploading if failed
                retry_cnt += 1
                self.logger.info("Retry Connection [{}/{}]...".format(retry_cnt, self.max_retry))
                self._connect_bucket()
        self.logger.info("Download failed!")

    def download_directory(self, oss_dir, local_dir, *args,
        force_overwrite_dir=False,
        **kwargs):
        # oss_prefix = os.path.join(self.personal_root, oss_dir) + "/"
        # for obj in oss2.ObjectIterator(self.bucket, prefix=oss_prefix):
        #     oss_path = os.path.relpath(obj.key, self.personal_root)
        #     local_path = os.path.join(local_dir, os.path.relpath(oss_path, oss_dir))
        #     self.download(oss_path, local_path)
        download_files = list(self._iter_oss_dir(oss_dir))
        if force_overwrite_dir:
            # TODO: this may be dangerous if downloading fails!
            # find local files that does not exist on oss and delete
            local_files = list(self._iter_local_dir(oss_dir))
            delete_files = set(local_files) - set(download_files)
            for fpath in delete_files:
                self._delete_local_by_relpath(oss_dir, fpath)

        # download
        if self.num_process > 0:
            new_obj = self._clone_wo_pool()
            for idx, _ in enumerate(
                self.pool.imap_unordered(functools.partial(new_obj._download_to_dir_by_relpath, oss_dir, local_dir), download_files)
            ):
                self.logger.info(f"Download process: {idx}/{len(download_files)}")
        else:
            for idx, filename in enumerate(download_files):
                self.logger.info(f"Download process: {idx}/{len(download_files)}")
                self._download_to_dir_by_relpath(oss_dir, local_dir, filename)

    def upload(self, oss_path, local_path, *args, checksum=True, allow_overwrite=True, **kwargs):
        if not os.path.isfile(local_path):
            self.logger.warning("local file {} not exist!".format(local_path))
            return
        oss_file = os.path.join(self.personal_root, oss_path)
        retry_cnt = 0
        while retry_cnt < self.max_retry or self.max_retry < 0:
            try:
                # check if object exists
                if self.bucket.object_exists(oss_file):
                    if not allow_overwrite:
                        self.logger.info("OSS File exists! Skip uploading oss file {} from {}".format(oss_path, local_path))
                        return
                    elif checksum and self._compare_checksum(oss_path, local_path):
                        self.logger.info("Same checksum! Skip uploading oss file {} from {}".format(oss_path, local_path))
                        return
                # uploading
                self.logger.info("Upload oss file {} from {}".format(oss_path, local_path))
                self.bucket.put_object_from_file(oss_file, local_path, *args, **kwargs)
                return
            except oss2.exceptions.RequestError:
                # retry uploading if failed
                retry_cnt += 1
                self.logger.info("Retry Connection [{}/{}]...".format(retry_cnt, self.max_retry))
                self._connect_bucket()
        self.logger.info("Upload failed!")

    # TODO: handle multiple client upload to the same directory case
    def upload_directory(self, oss_dir, local_dir, *args, 
        force_overwrite_dir=False,
        snapshot_local_dir=True, # snapshot dir to avoid local file change during upload
        **kwargs):
        if snapshot_local_dir:
            tmpdir = tempfile.TemporaryDirectory()
            # copy everything in local dir to tmp as snapshot
            snapshot_dir = os.path.join(tmpdir.name, "snapshot")
            shutil.copytree(local_dir, snapshot_dir)
            local_dir = snapshot_dir
        upload_files = list(self._iter_local_dir(local_dir))
        if force_overwrite_dir:
            # find oss files that does not exist on local and delete
            oss_files = list(self._iter_oss_dir(oss_dir))
            delete_files = set(oss_files) - set(upload_files)
            for fpath in delete_files:
                self._delete_oss_by_relpath(oss_dir, fpath)
        
        # upload
        if self.num_process > 0:
            new_obj = self._clone_wo_pool()
            for idx, _ in enumerate(
                self.pool.imap_unordered(functools.partial(new_obj._upload_to_dir_by_relpath, oss_dir, local_dir), upload_files)
            ):
                self.logger.info(f"Upload process: {idx}/{len(upload_files)}")
        else:
            for idx, filename in enumerate(upload_files):
                self.logger.info(f"Upload process: {idx}/{len(upload_files)}")
                self._upload_to_dir_by_relpath(oss_dir, local_dir, filename)

        # for dirpath, dirnames, filenames in os.walk(local_dir):
        #     for fname in filenames:
        #         local_path = os.path.join(dirpath, fname)
        #         oss_path = os.path.join(oss_dir, os.path.relpath(local_path, local_dir))
        #         self.upload(oss_path, local_path)

        if snapshot_local_dir:
            tmpdir.cleanup()

    # def download_archive_and_extract(self, oss_path, local_path, *args, **kwargs):
    #     self.download(oss_path, local_path, *args, **kwargs)
    #     extract_archive(local_path)

    def _diff_directory(self, oss_dir, local_dir, *args, **kwargs):
        upload_files, download_files = [], []

        # check files on oss and local
        local_files = list(self._iter_local_dir(local_dir))
        oss_files = list(self._iter_oss_dir(oss_dir))

        # diff 2 lists
        upload_files = set(local_files) - set(oss_files)
        download_files = set(oss_files) - set(local_files)

        return upload_files, download_files

    def sync_file(self, oss_path, local_path, *args, **kwargs):
        if os.path.exists(local_path):
            if not self.exists(oss_path):
                self.upload(oss_path, local_path, *args, **kwargs)
        else:
            self.download(oss_path, local_path, *args, **kwargs)

    def sync_directory(self, oss_dir, local_dir, *args,
        # check_hash=False,
        **kwargs):
        upload_files, download_files = self._diff_directory(
            oss_dir, local_dir, *args, **kwargs
        )

        # perform sync files
        if self.num_process > 0:
            new_obj = self._clone_wo_pool()
            for idx, _ in enumerate(
                self.pool.imap_unordered(functools.partial(new_obj._upload_to_dir_by_relpath, oss_dir, local_dir), upload_files)
            ):
                self.logger.info(f"Upload process: {idx}/{len(upload_files)}")
            for idx, _ in enumerate(
                self.pool.imap_unordered(functools.partial(new_obj._download_to_dir_by_relpath, oss_dir, local_dir), download_files)
            ):
                self.logger.info(f"Download process: {idx}/{len(download_files)}")
        else:
            for idx, filename in enumerate(upload_files):
                self.logger.info(f"Upload process: {idx}/{len(upload_files)}")
                self._upload_to_dir_by_relpath(oss_dir, local_dir, filename)
            for idx, filename in enumerate(download_files):
                self.logger.info(f"Download process: {idx}/{len(download_files)}")
                self._download_to_dir_by_relpath(oss_dir, local_dir, filename)


# import torch.utils.data
# class OSSDatasetWrapper(torch.utils.data._DatasetKind):
#     def __init__(self, dataset) -> None:
#         pass


# run this file to sync with oss
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="download")
    parser.add_argument("--oss_dir", '-o', type=str, default="experiments/aii_kube")
    parser.add_argument("--local_dir", '-l', type=str, default="experiments/aii_kube")
    parser.add_argument("--sync_dir", '-s', type=str)
    parser.add_argument("--allow_delete", '-d', action='store_true')
    parser.add_argument("--sync_interval", '-si', type=int, default=0)
    parser.add_argument("--num-process", '-p', type=int, default=0)

    args = parser.parse_args()

    # logger = setup_logger("OSSUtils", log_level='WARNING', log_level_file='INFO')
    # oss = OSSUtils(logger=logger)
    oss = OSSUtils(num_process=args.num_process)
    sync_local_dir = args.sync_dir if args.sync_dir else args.local_dir # "experiments/aii_kube"
    sync_oss_dir = args.sync_dir if args.sync_dir else args.oss_dir # "experiments/aii_kube"
    # oss.sync_directory(sync_oss_dir, sync_local_dir)
    while True:
        if args.op == "download":
            oss.download_directory(sync_oss_dir, sync_local_dir, force_overwrite_dir=args.allow_delete)
        elif args.op == "upload":
            oss.upload_directory(sync_oss_dir, sync_local_dir, force_overwrite_dir=args.allow_delete)
        elif args.op == "sync":
            oss.sync_directory(sync_oss_dir, sync_local_dir)
        else:
            raise KeyError(f"Unknown op {args.op}")

        if args.sync_interval > 0:
            print(f"Sleeping {args.sync_interval} seconds...")
            time.sleep(args.sync_interval)
        else:
            break