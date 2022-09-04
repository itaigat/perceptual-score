import torch

from torch import nn, Tensor
from torch.utils.data import DataLoader


def evaluate_risk(data_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, permute_first: bool = False) -> float:
    running_loss = 0.0
    data_points = 0

    for first_modality, second_modality, labels in data_loader:
        if permute_first:
            first_modality = first_modality[torch.randperm(first_modality.shape[0]), :]

        outputs = model(first_modality, second_modality)

        loss = loss_fn(outputs, labels).sum()

        running_loss += loss.detach().cpu().item()
        data_points += first_modality.shape[0]

    return running_loss / data_points


class TwoLayersFCNetwork(nn.Module):
    def __init__(self, input_size: int = 12, mid_dim: int = 7, output_size: int = 2):
        super(TwoLayersFCNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=mid_dim, out_features=output_size)
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        combined = torch.concat((x, y), dim=1)
        combined = self.network(combined)

        return combined
