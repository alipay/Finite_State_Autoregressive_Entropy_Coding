import pickle
import glob
import os

from .basic import MappingDataset, IterableDataset

# TODO: support other filesystems such as oss
class BinaryFilesDataset(MappingDataset):
    def __init__(self, *args,
                 files_glob="data/*",
                 packed_file=None,
                 segment_length=None,
                 discard_last=False,
                 **kwargs):
        self.file_list = glob.glob(files_glob)
        self.packed_file = packed_file
        self.segment_length = segment_length
        self.discard_last = discard_last

        # NOTE: if the packed_file is too big, out-of-memory may occur!
        if packed_file is not None:
            if os.path.exists(packed_file):
                with open(packed_file, 'rb') as f:
                    self.file_data = pickle.load(f)
                assert(isinstance(self.file_data, dict))
                self.file_list = list(self.file_data.keys())
                print(f"{len(self.file_list)} files loaded from {packed_file}")
            else:
                # pack a new file from the glob
                print(f"packing {len(self.file_list)} files to {packed_file}")
                self.file_data = dict()
                for file_path in self.file_list:
                    with open(file_path, 'rb') as f:
                        self.file_data[file_path] = f.read()
                with open(packed_file, 'wb') as f:
                    pickle.dump(self.file_data, f)
            self.file_lengths = [len(file_binary) for file_binary in self.file_data.values()]
        else:
            self.file_data = dict()
            if len(self.file_list) == 0:
                print("glob {} do not find any files!".format(files_glob))
            self.file_lengths = [os.path.getsize(file_path) for file_path in self.file_list]

        # find file seek ptrs
        if segment_length is None:
            self.file_seekers = [(fidx, 0) for fidx, _ in enumerate(self.file_list)]
        else:
            self.file_seekers = []
            file_ptr = 0
            seek_ptr = 0
            while file_ptr < len(self.file_lengths):
                if not (discard_last and seek_ptr + segment_length > self.file_lengths[file_ptr]):
                    self.file_seekers.append((file_ptr, seek_ptr))
                seek_ptr += segment_length
                if seek_ptr >= self.file_lengths[file_ptr]:
                    file_ptr += 1
                    seek_ptr = 0

    def __getitem__(self, index):
        file_ptr, seek_ptr = self.file_seekers[index]
        file_path = self.file_list[file_ptr]
        if file_path in self.file_data:
            if self.segment_length is None:
                return self.file_data[file_path]
            else:
                return self.file_data[file_path][seek_ptr:(seek_ptr+self.segment_length)]
        else:
            with open(file_path, 'rb') as f:
                f.seek(seek_ptr, 0)
                return f.read(self.segment_length)

    def __len__(self):
        return len(self.file_seekers)

    # def __iter__(self):
    #     for file_ptr, seek_ptr in self.file_seekers:
    #         with open(self.file_list[file_ptr], 'rb') as f:
    #             f.seek(seek_ptr, 0)
    #             byte_string = f.read(self.segment_length)
    #             yield byte_string


class BinaryStreamsDataset(IterableDataset):
    def __init__(self, file_streams, *args,
                 segment_length=None,
                 discard_last=False,
                 **kwargs):
        self.file_streams = file_streams
        self.segment_length = segment_length
        self.discard_last = discard_last

    def __iter__(self):
        for stream in self.file_streams:
            while True:
                byte_string = stream.read(self.segment_length)
                if len(byte_string) == self.segment_length:
                    yield byte_string
                else:
                    if not self.discard_last:
                        yield byte_string
                    break
    

if __name__ == "__main__":
    dataset = BinaryFilesDataset(files_glob="data/*/*", segment_length=16 * 1024)
    print(dataset.file_seekers)
    print(dataset[1])
    for i, bs in enumerate(dataset):
        print(i, bs)
