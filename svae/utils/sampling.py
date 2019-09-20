import torch


class Sampler:
    def __init__(self, temperature: float = 1.):
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.:
            logits.div_(self.temperature)
        return logits


class GreedySampler(Sampler):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.topk(logits, k=1, dim=-1)[1].squeeze(0)


class FullSampler(Sampler):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self._apply_temperature(logits)
        logp = torch.softmax(logits, dim=-1)
        idx_samples = logp.squeeze(0).multinomial(1)
        return idx_samples


class TopKSampler(FullSampler):
    def __init__(self, top_k: int = 10, temperature: float = 1.):
        if not top_k > 0:
            raise ValueError(f"Number of elements to sample from should be greater than zero.")
        self.top_k = top_k
        super(TopKSampler, self).__init__(temperature=temperature)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self._apply_temperature(logits)
        top_k_logits, top_k_indexes = logits.topk(k=self.top_k, dim=-1)
        indexes = torch.multinomial(top_k_logits, num_samples=1)
        sampled_indexes = top_k_indexes.gather(dim=1, index=indexes)
        return sampled_indexes


class TopPSampler(Sampler):
    def __init__(self, top_p: float):
        if not 0. < top_p <= 1.:
            raise ValueError(f"Wrong top_p value: {top_p}")
        self.top_p = top_p

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self._apply_temperature(logits)
        probs = logits.softmax(dim=-1)
        sorted_probs, sorted_indexes = probs.sort(dim=-1, descending=True)
        cumsum_probs = sorted_probs.cumsum(dim=-1)
        mask = cumsum_probs.lt(self.top_p)
        # add one more element to exceed top_p
        cumsum_mask = mask.cumsum(dim=-1)
        last_index = cumsum_mask[:, -1:]
        last_index.clamp_(0, mask.size(-1) - 1)
        mask.scatter_(-1, last_index, 1)

        # truncate elements
        trunc_dim = last_index.max()
        trunc_mask = mask[:, :trunc_dim + 1]
        trunc_probs = sorted_probs[:, :trunc_dim + 1]
        trunc_indexes = sorted_indexes[:, :trunc_dim + 1]

        # remove words that aren't in the mask
        words_to_remove = ~trunc_mask
        trunc_probs.masked_fill_(mask=words_to_remove, value=0.)

        # sample words
        indexes = trunc_probs.multinomial(num_samples=1)
        sampled_indexes = torch.gather(trunc_indexes, dim=-1, index=indexes)
        return sampled_indexes
