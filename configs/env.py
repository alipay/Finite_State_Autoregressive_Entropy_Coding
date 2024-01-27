'''
This file is for defining environment specific variables and its default value. 
Do not change this file directly!
Create a new env_config.py instead!
'''
from multiprocessing import cpu_count

# hardware
DEFAULT_CPU_CORES = cpu_count() # 0 means no multiprocessing
DEFAULT_GPU_DEVICES = -1 # could be like -1(all), 1, 4, [0, 1, 2, 3], "0, 1, 2, 3"
DEFAULT_MAX_MEMORY_CACHE = 0 # 0 mean no memory cache used

# paths
DEFAULT_PRETRAINED_PATH = "pretrained"
DEFAULT_EXPERIMENT_PATH = "experiments"
DEFAULT_DATA_PATH = "data"

# oss cloud config
DEFAULT_OSS_KEYID_BASE64 = ""
DEFAULT_OSS_KEYSEC_BASE64 = ""
DEFAULT_OSS_ENDPOINT = ""
DEFAULT_OSS_BUCKET_NAME = ""
DEFAULT_OSS_PERSONAL_ROOT = ""
DEFAULT_OSS_EXPERIMENT_PATH = "experiments"
DEFAULT_OSS_DATA_PATH = "data"

try:
    from .env_config import *
except ImportError:
    pass
