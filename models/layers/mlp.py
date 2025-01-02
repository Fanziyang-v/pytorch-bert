from torch import nn, Tensor


class MLP(nn.Module):
    """Multi-layer Perceptron layer."""

    def __init__(self, d_model: int, d_hidden: int, drop_prob: float = 0.1) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
