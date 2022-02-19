import torch

from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(self, n_samples: int = 1000, first_sample_size: int = 5, second_sample_size: int = 7):
        self.n_samples = n_samples
        self.first_sample_size = first_sample_size
        self.second_sample_size = second_sample_size

        self.first = torch.randn(self.n_samples, self.first_sample_size)
        self.second = torch.rand(self.n_samples, self.second_sample_size)
        self.labels = torch.randint(0, 2, (self.n_samples,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.first[idx], self.second[idx], self.labels[idx]


class PermutedMultiModalDataset(MultiModalDataset):
    def __init__(self, n_samples: int = 1000, first_sample_size: int = 5, second_sample_size: int = 7,
                 permute_first_batch: bool = False):
        super(PermutedMultiModalDataset, self).__init__(n_samples=n_samples,
                                                        first_sample_size=first_sample_size,
                                                        second_sample_size=second_sample_size)

        self.permute_first_batch = permute_first_batch

    def __getitem__(self, idx):
        if self.permute_first_batch:
            random_idx = torch.randint(0, self.__len__(), (1, )).item()
            first = self.first[random_idx]
        else:
            first = self.first[idx]

        return first, self.second[idx], self.labels[idx]
