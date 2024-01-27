import cbench.data.datasets
from configs.class_builder import ClassBuilder

config = ClassBuilder(
    cbench.data.datasets.BinaryFilesDataset,
    files_glob="data/enwik8/enwik8",
    discard_last=False
)