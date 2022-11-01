from block.datasets.neuromorphic import H5Dataset


class SSCDataset(H5Dataset):

    def __init__(self, root, train=True, dt=2, transform=None):
        super().__init__(root, train, n_in=700, n_out=35, t_len=500, train_name="ssc_train.h5", test_name="ssc_test.h5", dt=dt, transform=transform)