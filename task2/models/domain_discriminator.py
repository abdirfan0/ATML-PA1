
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_dimension=256,
        dropout_probability=0.5,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout_probability
            ),
            nn.Linear(
                hidden_dimension,
                2,
            ),
        )

    def forward(self, features):
        return self.network(features)
