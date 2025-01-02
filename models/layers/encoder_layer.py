from torch import nn, Tensor
from .mlp import MLP
from .attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
        drop_prob: float = 0.1,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, drop_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.mlp = MLP(d_model, d_hidden, drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        shortcut = x
        out = self.attn(x, x, x, mask)
        out = self.norm1(shortcut + self.dropout1(out))
        shortcut = out
        out = self.mlp(out)
        out = self.norm2(shortcut + self.dropout2(out))
        return out
