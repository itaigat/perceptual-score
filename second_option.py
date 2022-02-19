import torch.nn

from data import MultiModalDataset, PermutedMultiModalDataset
from torch.utils.data import DataLoader
from nn_utils import TwoLayersFCNetwork, evaluate_risk

if __name__ == '__main__':
    n_samples: int = 1000
    first_sample_size: int = 5
    second_sample_size: int = 7

    model = TwoLayersFCNetwork(input_size=first_sample_size + second_sample_size)

    dataset = MultiModalDataset(n_samples=n_samples,
                                first_sample_size=first_sample_size,
                                second_sample_size=second_sample_size)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    risk = evaluate_risk(data_loader=train_dataloader,
                         model=model,
                         loss_fn=loss_fn,
                         permute_first=True)

    dataset = PermutedMultiModalDataset(n_samples=n_samples,
                                        first_sample_size=first_sample_size,
                                        second_sample_size=second_sample_size)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    permuted_risk = evaluate_risk(data_loader=train_dataloader,
                                  model=model,
                                  loss_fn=loss_fn,
                                  permute_first=True)

    print(risk - permuted_risk)
