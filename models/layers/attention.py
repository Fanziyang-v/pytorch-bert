from torch import nn, Tensor
from functools import partial


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer."""

    def __init__(self, d_model: int, n_heads: int, drop_prob: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_o = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        q, k, v = map(partial(_split, n_heads=self.n_heads), (q, k, v))
        out = self.attn(q, k, v, mask)
        out = _concat(out)
        out = self.proj_o(out)
        return self.dropout(out)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention layer."""

    def __init__(self) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        d_k = q.size()[3]
        k_t = k.transpose(2, 3)
        scores: Tensor = (q @ k_t) * d_k**-0.5
        if mask is not None:
            scores.masked_fill_(mask, float("-inf"))
        attn = self.softmax(scores)
        out = attn @ v
        return out


def _split(tensor: Tensor, n_heads: int) -> Tensor:
    """Split tensor by number of heads."""
    batch_size, seq_len, d_model = tensor.size()
    d_head = d_model // n_heads
    return tensor.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)


def _concat(tensor: Tensor) -> Tensor:
    """Concatenate tensor by number of heads."""
    batch_size, n_heads, seq_len, d_head = tensor.size()
    d_model = n_heads * d_head
    return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
