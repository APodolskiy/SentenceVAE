from argparse import ArgumentParser
import json

from _jsonnet import evaluate_file

import torch
import torch.optim as optim
from torchtext.data import Field, Iterator

from svae.dataset_utils import *
from svae.dataset_utils.datasets import PTB
from svae.svae import SentenceVAE


if __name__ == '__main__':
    parser = ArgumentParser(description="Training of Sentence VAE")
    parser.add_argument("--config", type=str, required=False, metavar='PATH',
                        help="Path to a configuration file.")
    parser.add_argument("--run-dir", type=str, required=False, metavar='PATH',
                        help="Path to a directory where model "
                             "checkpoints will be stored.")
    args = parser.parse_args()

    #config = json.loads(evaluate_file(args.config))
    TEXT = Field(sequential=True, use_vocab=True, lwoer=True,
                 init_token=SOS_TOKEN, eos_token=EOS_TOKEN,
                 pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
                 tokenize=lambda x: x.strip().split(), include_lengths=True)
    fields = (('inp', TEXT), ('trg', TEXT))
    train_data, dev_data, test_data = PTB.splits(fields=fields, max_len=50)
    TEXT.build_vocab(train_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iter, dev_iter, test_iter = Iterator.splits(
        datasets=(train_data, dev_data, test_data),
        batch_sizes=(64, 100, 100),
        shuffle=True,
        sort_within_batch=True,
        sort_key=lambda x: len(x.inp),
        device=device
    )

    model = SentenceVAE(vocab=TEXT.vocab)
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    EPOCHS = 10
    for epoch in range(EPOCHS):
        # Training
        model.train()
        for batch in train_iter:
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            # TODO: add gradient clipping
            optimizer.step()
            print(f"Loss: {loss.item()}")
