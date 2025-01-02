from torch import nn, Tensor
from .layers.embedding import BERTEmbedding
from .encoder import TransformerEncoder


class BERT(nn.Module):
    """BERT model(encoder only)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_hidden: int,
        max_len: int,
        n_layers: int,
        n_heads: int,
        num_segments: int = 2,
        drop_prob: float = 0.1,
    ) -> None:
        super(BERT, self).__init__()
        # embedding layer
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            num_segments=num_segments,
            drop_prob=drop_prob,
        )
        # transformer encoder
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            d_model=d_model,
            d_hidden=d_hidden,
            n_heads=n_heads,
            drop_prob=drop_prob,
        )

    def forward(self, x: Tensor, segment_label: Tensor, mask: Tensor) -> Tensor:
        x = self.embedding(x, segment_label)
        return self.encoder(x, mask)


def bert_base(vocab_size: int) -> BERT:
    return BERT(
        vocab_size=vocab_size,
        d_model=768,
        d_hidden=3072,
        max_len=512,
        n_layers=12,
        n_heads=12,
        num_segments=2,
        drop_prob=0.1,
    )


def bert_large(vocab_size: int) -> BERT:
    return BERT(
        vocab_size=vocab_size,
        d_model=1024,
        d_hidden=4096,
        max_len=512,
        n_layers=24,
        n_heads=16,
        num_segments=2,
        drop_prob=0.1,
    )
