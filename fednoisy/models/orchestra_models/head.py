import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128):
        super(LinearHead, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.linear(x)