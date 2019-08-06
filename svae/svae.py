from typing import List

import torch
import torch.nn as nn
from torchtext.vocab import Vocab

from svae.encoder import RNNEncoder
from svae.decoder import RNNDecoder


class SentenceVAE(nn.Module):
    def __init__(self, vocab: Vocab,
                 emb_dim: int = 128,
                 word_drop_p: float = 0.2,
                 tie_weights: bool = False):
        super().__init__()
        self.emb_dim = emb_dim
        self.word_drop_p = word_drop_p
        # Special symbols
        self.unk_idx = vocab.stoi['<unk>']
        self.pad_idx = vocab.stoi['<pad>']
        self.sos_idx = vocab.stoi['<s>']
        self.eos_idx = vocab.stoi['</s>']
        # Model
        self.embedding = nn.Embedding(len(vocab), self.emb_dim)
        self.encoder = RNNEncoder(input_size=emb_dim, pad_value=self.pad_idx)
        self.decoder = RNNDecoder(input_size=emb_dim, pad_value=self.pad_idx)
        self.code2mu = nn.Linear(128, 128)
        self.code2sigma = nn.Linear(128, 128)
        self.out2vocab = nn.Linear(128, len(vocab))
        # Tie weights
        if tie_weights:
            self.out2vocab.weight = self.embedding.weight

        self.loss_func = nn.NLLLoss(ignore_index=self.pad_idx, reduction='sum')
        #self.loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='sum')

    def forward(self, batch):
        inp, inp_lengths = batch.inp
        trg, trg_lengths = batch.trg
        batch_size = inp.size(0)
        inp_emb = self.embedding(inp)
        enc = self.encoder(inp_emb, inp_lengths)

        mu = self.code2mu(enc)
        log_sigma = self.code2sigma(enc)
        kl_loss = 0.5 * torch.sum(2*log_sigma + log_sigma.exp() + mu.pow(2) - 1)

        sigma = torch.exp(log_sigma)
        z = self.sample_z(mu, sigma)

        trg_inp, trg_out = trg[:-1], trg[1:]
        trg_inp = self.word_dropout(trg_inp)
        trg_emb = self.embedding(trg_inp)
        out = self.decoder(trg_emb, trg_lengths, z)
        logits = self.out2vocab(out)
        logp = torch.log_softmax(logits, dim=-1)
        loss_xe = self.loss_func(logp.transpose(0, 1).contiguous().view(batch_size, -1), trg_inp)
        # TODO: change loss computation
        loss_xe = loss_xe.mean()

        loss = kl_loss + loss_xe
        return loss

    def sample_z(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        z = eps * sigma + mu
        return z

    def word_dropout(self, inp: torch.Tensor) -> torch.Tensor:
        tokens = inp.clone()
        drop_probs = torch.rand_like(tokens)
        drop_probs[(tokens == self.sos_idx) | (tokens == self.eos_idx) | (tokens == self.pad_idx)] = 1
        mask = drop_probs < self.word_drop_p
        tokens[mask] = self.unk_idx
        tokens = tokens.to(inp.device)
        return tokens

    def sample(self, num_samples: int) -> List[str]:
        pass
