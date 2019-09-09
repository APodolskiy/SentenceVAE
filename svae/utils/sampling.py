import torch


class Sampler:
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplemented


class GreedySampler(Sampler):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.topk(logits, k=1, dim=-1)[1].squeeze(0)


class FullSampler(Sampler):
    def __init__(self, temperature: float = 1.):
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.:
            logits.div_(self.temperature)
        logp = torch.softmax(logits, dim=-1)
        idx_samples = logp.squeeze(0).multinomial(1)
        return idx_samples


class TopKSampler(Sampler):
    def __init__(self, top_k: int = 10):
        if not top_k > 0:
            raise ValueError(f"Number of elements to sample from should be greater than zero.")
        self.top_k = top_k

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        pass


class TopPSampler(Sampler):
    def __init__(self, top_p: float):
        if not 0. < top_p <= 1.:
            raise ValueError(f"Wrong top_p value: {top_p}")
        self.top_p = top_p

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        pass
