import torch as T
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, fcs: tuple[int]) -> None:
        super().__init__()
        self.fcs = fcs
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.layers_dim = [input_dim + action_dim, *fcs, 1]
        self.model = self.get_model()

    def forward(self, state: T.Tensor, action: T.Tensor):
        x = T.cat([state, action], dim=1)
        return self.model(x)

    def get_model(self):
        layers = []
        for i in range(len(self.layers_dim) - 1):
            layers.append(nn.Linear(self.layers_dim[i], self.layers_dim[i + 1]))
            layers.append(nn.ReLU())
        model = nn.Sequential(*layers)
        return model
