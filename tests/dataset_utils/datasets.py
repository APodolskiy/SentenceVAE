import pytest

from torchtext.data import Field

from svae.dataset_utils import SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN
from svae.dataset_utils.datasets import YelpReview


def test_yelp_review():
    TEXT = Field(sequential=True, use_vocab=True, lower=True, init_token=SOS_TOKEN, eos_token=EOS_TOKEN,
                 pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, tokenize="spacy", include_lengths=True)
    dataset = YelpReview('data/yelp_review/train.csv', fields=(('inp', TEXT), ('trg', TEXT)), num_samples=120_000)
    train, dev, test = dataset.split(split_ratio=[100_000, 10_000, 10_000])
    assert len(train) == 100_000
    assert len(dev) == 10_000
    assert len(test) == 10_000
