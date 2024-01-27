import glob
import os
import pandas as pd
import numpy as np

# from .basic import MappingDataset, IterableDataset


class PandasTableBaseDataset(object):
    def __init__(self, *args,
                 files_glob: str = "data/*",
                 serialize_format: str = "csv",
                 serialize_config: dict = None,
                 **kwargs):
        self.file_list = glob.glob(files_glob)
        if len(self.file_list) == 0:
            print("glob {} do not find any files!".format(files_glob))
        else:
            print("{} files found!".format(len(self.file_list)))

        self.serialize_format = serialize_format
        # TODO: default configs for different formats
        if serialize_config is None:
            serialize_config = dict(index=False)
        self.serialize_config = serialize_config

    def _load_table(self, file_name):
        if file_name.endswith(".csv"):
            return pd.read_csv(file_name)
        elif file_name.endswith(".npy"):
            return pd.DataFrame.from_records(np.load(file_name, allow_pickle=True))
        # TODO: other formats

    def _serialize_table(self, data_frame):
        if self.serialize_format == "csv":
            # csv serialization
            return data_frame.to_csv(**self.serialize_config).encode('utf-8')
        elif self.serialize_format == "json":
            return data_frame.to_json(**self.serialize_config).encode('utf-8')


class PandasTableMappingDataset(PandasTableBaseDataset):
    def __getitem__(self, index) -> pd.DataFrame:
        table_data = self._load_table(self.file_list[index])
        return table_data

    def __len__(self):
        return len(self.file_list)


class PandasTableStreamDataset(PandasTableBaseDataset):
    def __init__(self, *args,
                 fetch_mode="row",  # row/column
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fetch_mode = fetch_mode

    def __iter__(self):
        for file_name in self.file_list:
            table_data = self._load_table(file_name)
            # TODO: output format
            if self.fetch_mode == "row":
                for label, content in table_data.iterrows():
                    yield self._serialize_table(content)  # content.to_csv(index=False, header=False)
            else:
                for label, content in table_data.iteritems():
                    yield self._serialize_table(content)  # content.to_csv(index=False, header=False)


if __name__ == "__main__":
    # dataset = PandasTableDataset(files_glob="data/*/*.csv")
    dataset = PandasTableStreamDataset(files_glob="data/*.csv", fetch_mode="row")
    for i, bs in enumerate(dataset):
        print(i, bs)
        break
