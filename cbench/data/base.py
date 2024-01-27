import abc


class DataLoaderInterface(object):
    @abc.abstractmethod
    def get_length(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def iterate(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_at(self, index):
        raise NotImplementedError()

    def __len__(self):
        return self.get_length()

    def __getitem__(self, index):
        return self.get_data_at(index)

    def __iter__(self):
        return self.iterate()

# TODO: datasets may implement __getitem__, __len__ or __iter__. Find a proper API for this definition!
# class DatasetInterface(object):
#     @abc.abstractmethod
#     def get_length(self):
#         raise NotImplementedError()

#     @abc.abstractmethod
#     def get_data_at(self, index):
#         raise NotImplementedError()
