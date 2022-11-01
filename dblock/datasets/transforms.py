import brainbox
from block.datasets.transforms import SpikeTensorBuilder


class List:

    @staticmethod
    def get_nmnist_transform(t_len, use_augmentation=False):
        if use_augmentation:
            raise NotImplementedError
        else:
            transform_list = [SpikeTensorBuilder(n_units=1156, t_len=t_len, dt=1)]

        return brainbox.datasets.transforms.Compose(transform_list)

    @staticmethod
    def get_shd_transform(t_len, use_augmentation=False):
        if use_augmentation:
            raise NotImplementedError
        else:
            transform_list = [SpikeTensorBuilder(n_units=700, t_len=t_len, dt=2)]

        return brainbox.datasets.transforms.Compose(transform_list)

    @staticmethod
    def get_ssc_transform(t_len, use_augmentation=False):
        if use_augmentation:
            raise NotImplementedError
        else:
            transform_list = [SpikeTensorBuilder(n_units=700, t_len=t_len, dt=2)]

        return brainbox.datasets.transforms.Compose(transform_list)
