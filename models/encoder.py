from torch import nn, Tensor
from .layers.encoder_layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    """Transformer Encoder."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_hidden: int,
        n_heads: int,
        drop_prob: float = 0.1,
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_hidden, drop_prob)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        return out

