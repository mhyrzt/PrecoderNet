import torch as T
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, fcs: tuple[int]) -> None:
        super().__init__()
        self.fcs = fcs
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.layers_dim = [input_dim, *fcs, action_dim]
        self.model = self.get_model()

    def forward(self, state: T.Tensor):
        return self.model(state)

    def get_model(self):
        layers = []
        for i in range(len(self.layers_dim)- 1):
            inp = self.layers_dim[i]
            out = self.layers_dim[i + 1]
            layers.append(nn.Linear(inp, out))
            layers.append(nn.ReLU())
        model = nn.Sequential(*layers)
        return model
