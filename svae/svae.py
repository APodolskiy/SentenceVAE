from typing import List

import torch
import torch.nn as nn
from torchtext.vocab import Vocab

from svae.encoder import RNNEncoder
from svae.decoder import RNNDecoder
from svae.dataset_utils import *
from svae.utils.annealing import SigmoidAnnealing


class SentenceVAE(nn.Module):
    def __init__(self,
                 vocab: Vocab,
                 emb_dim: int = 128,
                 word_drop_p: float = 0.2,
                 tie_weights: bool = False):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_size = 128
        self.word_drop_p = word_drop_p
        # Special symbols
        self.unk_idx = vocab.stoi[UNK_TOKEN]
        self.pad_idx = vocab.stoi[PAD_TOKEN]
        self.sos_idx = vocab.stoi[SOS_TOKEN]
        self.eos_idx = vocab.stoi[EOS_TOKEN]
        # Model
        self.embedding = nn.Embedding(len(vocab), self.emb_dim)
        self.encoder = RNNEncoder(input_size=self.emb_dim, pad_value=self.pad_idx)
        self.decoder = RNNDecoder(input_size=self.emb_dim, pad_value=self.pad_idx)
        self.code2mu = nn.Linear(2*128, self.latent_size)
        self.code2sigma = nn.Linear(2*128, self.latent_size)
        self.out2vocab = nn.Linear(128, len(vocab))
        # Tie weights
        if tie_weights:
            self.out2vocab.weight = self.embedding.weight

        self.annealing_function = SigmoidAnnealing()
        self.loss_func = nn.NLLLoss(ignore_index=self.pad_idx, reduction='none')

    def forward(self, batch):
        inp, inp_lengths = batch.inp
        trg, trg_lengths = batch.trg
        seq_len, batch_size = inp.size()
        inp_emb = self.embedding(inp)
        enc = self.encoder(inp_emb, inp_lengths)

        mu = self.code2mu(enc)
        log_sigma = self.code2sigma(enc)
        kl_loss = 0.5 * torch.sum(log_sigma.exp() + mu.pow(2) - 1 - log_sigma)

        sigma = torch.exp(0.5*log_sigma)
        z = self.sample_posterior(mu, sigma)

        trg_inp, trg_out = trg[:-1], trg[1:]
        trg_inp = self.word_dropout(trg_inp)
        trg_emb = self.embedding(trg_inp)
        out = self.decoder(trg_emb, trg_lengths - 1, z)
        logits = self.out2vocab(out)
        logp = torch.log_softmax(logits, dim=-1)
        logp_words = logp.transpose(0, 1).contiguous().view(batch_size*(seq_len - 1), -1)
        target = trg_out.transpose(0, 1).contiguous().view(-1)
        loss_xe = self.loss_func(logp_words, target)
        # TODO: mean along sentence or sum whole losses
        loss_xe = loss_xe.view(batch_size, seq_len - 1).mean(dim=1).sum()

        kl_coeff = self.annealing_function()
        loss = loss_xe + kl_coeff * kl_loss
        return loss

    def sample_posterior(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu).to(sigma.device)
        z = eps * sigma + mu
        return z

    def sample_prior(self, num_samples: int):
        return torch.randn(num_samples, self.latent_size)

    def word_dropout(self, inp: torch.Tensor) -> torch.Tensor:
        tokens = inp.clone()
        drop_probs = torch.rand(*tokens.size())
        drop_probs[(tokens == self.sos_idx) | (tokens == self.eos_idx) | (tokens == self.pad_idx)] = 1
        mask = drop_probs < self.word_drop_p
        tokens[mask] = self.unk_idx
        tokens = tokens.to(inp.device)
        return tokens

    def sample(self, num_samples: int, max_len: int = 50) -> List[str]:
        pass
