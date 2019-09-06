from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torchtext.vocab import Vocab

from svae.encoder import RNNEncoder
from svae.decoder import RNNDecoder
from svae.dataset_utils import *
from svae.utils.annealing import LogisticAnnealing, LinearAnnealing
from svae.utils.training import AverageMetric, Params


class RecurrentVAE(nn.Module):
    def __init__(self,
                 vocab: Vocab,
                 params: Params):
        super().__init__()
        self.vocab = vocab
        self.params = params
        self.embed_dim = params.embed_dim
        self.latent_dim = params.latent_dim
        self.word_drop_p = params.word_drop_p
        self.greedy = params.get('greedy', False)
        # Special symbols
        self.unk_idx = self.vocab.stoi[UNK_TOKEN]
        self.pad_idx = self.vocab.stoi[PAD_TOKEN]
        self.sos_idx = self.vocab.stoi[SOS_TOKEN]
        self.eos_idx = self.vocab.stoi[EOS_TOKEN]
        # Model
        self.embedding = nn.Embedding(len(self.vocab), self.embed_dim)
        self.embed_drop = nn.Dropout(self.params.get('embedding_drop_p', 0.0))

        encoder_params = params.pop('encoder')
        decoder_params = params.pop('decoder')
        self.encoder = RNNEncoder(**encoder_params, pad_value=self.pad_idx)
        self.decoder = RNNDecoder(**decoder_params, pad_value=self.pad_idx)

        self.code2mu = nn.Linear(params.hidden_output_size, self.latent_dim)
        self.code2sigma = nn.Linear(params.hidden_output_size, self.latent_dim)
        self.latent2hidden = nn.Linear(self.latent_dim, decoder_params.hidden_size)
        # Tie weights
        self.out2vocab = nn.Sequential()
        if params.tie_weights:
            project_hidden = nn.Linear(decoder_params.hidden_size, self.embed_dim)
            out2vocab = nn.Linear(self.embed_dim, len(self.vocab))
            out2vocab.weight = self.embedding.weight
            self.out2vocab.add_module('projection', project_hidden)
            self.out2vocab.add_module('out', out2vocab)
        else:
            out2vocab = nn.Linear(decoder_params.hidden_size, len(self.vocab))
            self.out2vocab.add_module('out', out2vocab)

        annealing_params = self.params.pop('annealing')
        annealing_type = annealing_params.pop('type')
        if annealing_type == 'logistic':
            self.annealing_function = LogisticAnnealing(**annealing_params)
        elif annealing_type == 'linear':
            self.annealing_function = LinearAnnealing(**annealing_params)
        else:
            raise NotImplemented(f"Annealing function of type: {annealing_type} is not implemented.")

        self.loss_func = nn.NLLLoss(ignore_index=self.pad_idx, reduction='none')

        self.elbo_metric = AverageMetric()
        self.rec_loss_metric = AverageMetric()
        self.kl_loss_metric = AverageMetric()

    def forward(self, batch):
        inp, inp_lengths = batch.inp
        trg, trg_lengths = batch.trg
        seq_len, batch_size = inp.size()

        encoder_output = self.encode(inp, inp_lengths)
        z, kl_loss = encoder_output['code'], encoder_output['kl_loss']
        self.kl_loss_metric(kl_loss.item() * batch_size, num_steps=batch_size)

        trg_inp, trg_out = trg[:-1], trg[1:]
        out, _ = self.decode(z, trg_inp, trg_lengths - 1)

        logits = self.out2vocab(out)
        logp = torch.log_softmax(logits, dim=-1)
        logp_words = logp.transpose(0, 1).contiguous().view(-1, len(self.vocab))
        target = trg_out.transpose(0, 1).contiguous().view(-1)
        loss_xe = self.loss_func(logp_words, target)
        loss_xe = loss_xe.view(batch_size, seq_len - 1).sum(dim=1).mean()
        self.rec_loss_metric(loss_xe.item() * batch_size, num_steps=batch_size)

        kl_coeff = self.annealing_function()
        loss = loss_xe + kl_coeff * kl_loss
        self.elbo_metric((-loss_xe - kl_loss).item() * batch_size, num_steps=batch_size)
        return {
            'loss': loss,
            'rec_loss': loss_xe.item(),
            'kl_loss': kl_loss.item(),
            'kl_weight': kl_coeff
        }

    def encode(self, inp: torch.Tensor, inp_lengths: torch.Tensor, use_mean: bool = False) -> Dict[str, torch.Tensor]:
        _, batch_size = inp.size()
        inp_emb = self.embedding(inp)
        inp_emb = self.embed_drop(inp_emb)
        enc = self.encoder(inp_emb, inp_lengths)

        mu = self.code2mu(enc)
        log_sigma = self.code2sigma(enc)
        kl_loss = 0.5 * torch.sum(log_sigma.exp() + mu.pow(2) - 1 - log_sigma, dim=1).mean()

        if not use_mean:
            sigma = torch.exp(0.5 * log_sigma)
            z = self.sample_posterior(mu, sigma)
        else:
            z = mu
        return {
            'code': z,
            'kl_loss': kl_loss
        }

    def decode(self, latent: torch.Tensor,
               trg_inp: torch.Tensor,
               trg_lengths: torch.Tensor) -> torch.Tensor:
        h_init = self.latent2hidden(latent)
        h_init = h_init.expand((self.decoder.num_layers, *h_init.size())).contiguous()
        if self.decoder.type == 'lstm':
            h_init = (h_init, torch.zeros_like(h_init))
        trg_inp = self.word_dropout(trg_inp)
        trg_emb = self.embedding(trg_inp)
        trg_emb = self.embed_drop(trg_emb)
        out = self.decoder(trg_emb, trg_lengths, h_init)
        return out

    def sample_posterior(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu).to(sigma.device)
        z = eps * sigma + mu
        return z

    def sample_prior(self, num_samples: int):
        return torch.randn(num_samples, self.latent_dim)

    def word_dropout(self, inp: torch.Tensor) -> torch.Tensor:
        tokens = inp.clone()
        drop_probs = torch.rand(*tokens.size())
        drop_probs[(tokens == self.sos_idx) | (tokens == self.eos_idx) | (tokens == self.pad_idx)] = 1
        mask = drop_probs < self.word_drop_p
        tokens[mask] = self.unk_idx
        tokens = tokens.to(inp.device)
        return tokens

    def get_metrics(self, reset: bool = False):
        return {
            'elbo': self.elbo_metric.get_metric(reset),
            'data_term': self.rec_loss_metric.get_metric(reset),
            'kl_term': self.kl_loss_metric.get_metric(reset)
        }

    def sample(self,
               z: Optional[torch.tensor] = None,
               num_samples: Optional[int] = None,
               max_len: int = 60,
               device=torch.device('cpu')) -> List[str]:
        if z is None:
            z = self.sample_prior(num_samples)
        z = z.to(device)
        num_samples = z.size(0)

        prev_words = torch.tensor([[self.sos_idx]*num_samples])
        prev_words = prev_words.to(device)
        gen_words = []

        hidden = self.latent2hidden(z)
        hidden = hidden.expand((self.decoder.num_layers, *hidden.size())).contiguous()
        if self.decoder.type == 'lstm':
            hidden = (hidden, torch.zeros_like(hidden))

        done = [False] * num_samples
        lengths = torch.tensor([1]*num_samples)

        iters = 0
        with torch.no_grad():
            while not all(done) and iters < max_len:
                inp_embed = self.embedding(prev_words)
                out, hidden = self.decoder(inp_embed, lengths, hidden)
                logits = self.out2vocab(out)
                prev_words = self._sample_words(logits)
                done = [(el[0] == self.eos_idx) and flag for el, flag in zip(prev_words, done)]
                prev_words = prev_words.t()
                gen_words.append(prev_words)
                iters += 1
        # sentences acquisition
        indices_seqs = torch.cat(gen_words, dim=0).t().tolist()
        samples = []
        for indices in indices_seqs:
            sentence = self._idx2sentence(indices)
            samples.append(sentence)
        return samples

    def _sample_words(self, logits: torch.Tensor) -> torch.Tensor:
        if self.greedy:
            return torch.topk(logits, k=1, dim=-1)[1].squeeze(0)
        logp = torch.softmax(logits, dim=-1)
        idx_samples = logp.squeeze(0).multinomial(1)
        return idx_samples

    def _idx2sentence(self, indices: List[int]) -> str:
        tokens = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            tokens.append(self.vocab.itos[idx])
        return ' '.join(tokens)
