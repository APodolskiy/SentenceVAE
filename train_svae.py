from argparse import ArgumentParser
import json
from pprint import pprint
import shutil

from _jsonnet import evaluate_file
from pathlib import Path

from tensorboardX import SummaryWriter
from tqdm import tqdm

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
                        help="Path to a directory where model checkpoints will be stored.")
    parser.add_argument("--force", action='store_true',
                        help="Whether to rewrite data if run directory already exists.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.exists() and args.force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    writer = SummaryWriter(logdir=str(run_dir))

    #config = json.loads(evaluate_file(args.config))
    TEXT = Field(sequential=True, use_vocab=True, lower=True,
                 init_token=SOS_TOKEN, eos_token=EOS_TOKEN,
                 pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
                 tokenize=lambda x: x.strip().split(), include_lengths=True)
    fields = (('inp', TEXT), ('trg', TEXT))
    train_data, dev_data, test_data = PTB.splits(fields=fields)
    TEXT.build_vocab(train_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    train_iter, dev_iter, test_iter = Iterator.splits(
        datasets=(train_data, dev_data, test_data),
        batch_sizes=(32, 100, 100),
        shuffle=True,
        sort_within_batch=True,
        sort_key=lambda x: len(x.inp),
        device=device
    )

    model = SentenceVAE(vocab=TEXT.vocab)
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    EPOCHS = 10
    iters = 0
    for epoch in range(EPOCHS):
        print("#"*20)
        print(f"EPOCH {epoch}\n")
        # Training
        model.train()
        for batch in tqdm(train_iter):
            iters += 1
            output = model(batch)
            loss = output['loss']
            optimizer.zero_grad()
            loss.backward()
            # TODO: add gradient clipping
            optimizer.step()
            writer.add_scalar('train/ELBO', -loss.item(), iters)
            writer.add_scalar('train/rec_loss', output['rec_loss'], iters)
            writer.add_scalar('train/kl_loss', output['kl_loss'], iters)
            writer.add_scalar('train/kl_weight', output['kl_weight'], iters)
        metrics = model.get_metrics(reset=True)
        for metric, value in metrics.items():
            writer.add_scalar(f'train/{metric}', value, epoch)
        # Validation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dev_iter):
                output = model(batch)
            metrics = model.get_metrics(reset=True)
            for metric, value in metrics.items():
                writer.add_scalar(f'dev/{metric}', value, epoch)

        print("Sentence samples.")
        samples = model.sample(num_samples=10, device=device)
        pprint(samples)

    # TODO: model saving
    writer.close()
