import torch
from torch import nn, Tensor
from .bert import BERT


class BERTLanguageModeling(nn.Module):
    def __init__(self, bert: BERT) -> None:
        super(BERTLanguageModeling, self).__init__()
        self.bert = bert
        self.masked_lm = MaskedLanguageModeling(bert.d_model, bert.vocab_size)
        self.next_sent_pred = NextSentencePrediction(bert.d_model)

    def forward(
        self, x: Tensor, segment_label: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        x = self.bert(x, segment_label, mask)
        return self.masked_lm(x), self.next_sent_pred(x)


class MaskedLanguageModeling(nn.Module):
    """Masked Language Modeling head for BERT."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(MaskedLanguageModeling, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        out = self.softmax(out)
        return out


class NextSentencePrediction(nn.Module):
    """Next Sentence Prediction head for BERT."""

    def __init__(self, d_model: int) -> None:
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(d_model, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        out = self.softmax(out)
        return out
