from argparse import ArgumentParser
import json
from pprint import pprint
import shutil
from typing import Dict, Optional, Callable

from _jsonnet import evaluate_file
from pathlib import Path

import dill
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.optim as optim
from torchtext.data import Field, Iterator

from svae.dataset_utils import *
from svae.dataset_utils.datasets import PTB, YelpReview
from svae.svae import RecurrentVAE
from svae.utils.scheduler import WarmUpDecayLR
from svae.utils.training import save_checkpoint, Params


def train(train_dir: str, config: Dict, force: bool = False, metric_logger: Optional[Callable] = None):
    train_dir = Path(train_dir)
    if train_dir.exists() and force:
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=False)

    params_file = train_dir / f"config.jsonnet"
    with params_file.open('w') as fp:
        json.dump(config, fp)
    params = Params(config)
    pprint(f"Config:")
    pprint(config)

    writer = SummaryWriter(logdir=str(train_dir))

    training_params = params.pop('training')
    dataset_params = params.pop('dataset')
    sampling_params = params.pop('sampling')
    sampling_temperatures = sampling_params.get('temperature', [1.0])
    if isinstance(sampling_temperatures, (int, float)):
        sampling_temperatures = [sampling_temperatures]

    dataset_name = dataset_params.pop('name', "PTB")
    # TODO: unify datasets creation
    if dataset_name == "PTB":
        TEXT = Field(sequential=True, use_vocab=True, lower=True,
                     init_token=SOS_TOKEN, eos_token=EOS_TOKEN,
                     pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
                     tokenize=lambda x: x.strip().split(), include_lengths=True)
        fields = (('inp', TEXT), ('trg', TEXT))
        train_data, dev_data, test_data = PTB.splits(fields=fields)
    elif dataset_name == "YelpReview":
        TEXT = Field(sequential=True, use_vocab=True, lower=True,
                     init_token=SOS_TOKEN, eos_token=EOS_TOKEN,
                     pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
                     tokenize="spacy", include_lengths=True)
        fields = (('inp', TEXT), ('trg', TEXT))
        train_data, dev_data, test_data = YelpReview.splits(fields=fields,
                                                            num_samples=120_000,
                                                            split_ratio=[100_000, 10_000, 10_000],
                                                            max_len=150)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported!")

    TEXT.build_vocab(train_data, max_size=20_000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    train_iter, dev_iter, test_iter = Iterator.splits(
        datasets=(train_data, dev_data, test_data),
        batch_sizes=(training_params.batch_size,
                     training_params.test_batch_size,
                     training_params.test_batch_size),
        shuffle=True,
        sort_within_batch=True,
        sort_key=lambda x: len(x.inp),
        device=device
    )

    model = RecurrentVAE(vocab=TEXT.vocab, params=params.pop('model'))
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), **training_params.pop('optimizer'))

    scheduler = None
    scheduler_params = training_params.pop('lr_scheduler', None)
    if scheduler_params is not None:
        scheduler = WarmUpDecayLR(optimizer=optimizer, **scheduler_params)

    iters = 0
    for epoch in range(training_params.epochs):
        print("#" * 20)
        print(f"EPOCH {epoch}\n")
        # Training
        model.train()
        for batch in tqdm(train_iter, desc='Training'):
            iters += 1
            output = model(batch)
            loss = output['loss']
            optimizer.zero_grad()
            loss.backward()
            # TODO: add gradient clipping
            optimizer.step()
            writer.add_scalar('train/ELBO', -output['rec_loss'] - output['kl_loss'], iters)
            writer.add_scalar('train/rec_loss', output['rec_loss'], iters)
            writer.add_scalar('train/kl_loss', output['kl_loss'], iters)
            writer.add_scalar('train/kl_weight', output['kl_weight'], iters)
        metrics = model.get_metrics(reset=True)
        for metric, value in metrics.items():
            writer.add_scalar(f'train/{metric}', value, epoch)
        if metric_logger is not None:
            metric_logger({f"train_{key}": value for key, value in metrics.items()}, epoch)
        # Validation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dev_iter, desc='Validation'):
                _ = model(batch)
            valid_metrics = model.get_metrics(reset=True)
            for metric, value in valid_metrics.items():
                writer.add_scalar(f'dev/{metric}', value, epoch)
            if metric_logger is not None:
                metric_logger({f"valid_{key}": value for key, value in valid_metrics.items()}, epoch)

        for temperature in sampling_temperatures:
            print("#" * 20)
            print(f"Sentence samples. Temperature: {temperature}")
            samples = model.sample(num_samples=10,
                                   temperature=temperature,
                                   device=device,
                                   max_len=sampling_params.get('max_len', 50))
            print(*samples, sep='\n')
        if scheduler_params is not None:
            scheduler.step()

    with (train_dir / 'TEXT.Field').open("wb") as fp:
        dill.dump(TEXT, fp)
    save_checkpoint(model.state_dict(), train_dir)

    if params.get('eval_on_test', False):
        print("Evaluating model on test data...")
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_iter, desc='Test set evaluation'):
                _ = model(batch)
            test_metrics = model.get_metrics(reset=True)
            for metric, value in test_metrics.items():
                print(f"{metric}: {value}")
            if metric_logger is not None:
                metric_logger(test_metrics)

    writer.close()


if __name__ == '__main__':
    parser = ArgumentParser(description="Training of Sentence VAE")
    parser.add_argument("--config", type=str, required=True, metavar='PATH',
                        help="Path to a configuration file.")
    parser.add_argument("--run-dir", type=str, required=True, metavar='PATH',
                        help="Path to a directory where model checkpoints will be stored.")
    parser.add_argument("--force", action='store_true',
                        help="Whether to rewrite data if run directory already exists.")
    args = parser.parse_args()

    config = json.loads(evaluate_file(args.config))
    train(args.run_dir, config, args.force)
