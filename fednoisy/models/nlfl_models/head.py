import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)