import torch
from torch import nn, Tensor


class BERTEmbedding(nn.Module):
    """BERT Embedding layer containing three embeddings:

    1. Token Embedding
    2. Position Embedding
    3. Segment Embedding
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        num_segments: int = 2,
        drop_prob: float = 0.1,
    ) -> None:
        super(BERTEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, max_len)
        self.seg_emb = SegmentEmbedding(num_segments, d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor, segment_label: Tensor) -> Tensor:
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        seg_emb = self.seg_emb(segment_label)
        return self.dropout(tok_emb + pos_emb + seg_emb)


class TokenEmbedding(nn.Embedding):
    """Token Embedding layer."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=0)


class PositionEmbedding(nn.Module):
    """Position Embedding layer."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super(PositionEmbedding, self).__init__()
        self.pos_encoding = torch.zeros(max_len, d_model, requires_grad=False)
        factor = 10000 ** (torch.arange(0, d_model, step=2) / d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        self.pos_encoding[:, 0::2] = torch.sin(pos / factor)
        self.pos_encoding[:, 1::2] = torch.cos(pos / factor)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size()[1]
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).to(x.device)
        return pos_emb


class SegmentEmbedding(nn.Embedding):
    """Segment Embedding layer."""

    def __init__(self, num_segments: int, d_model: int) -> Tensor:
        super(SegmentEmbedding, self).__init__(num_segments, d_model)
