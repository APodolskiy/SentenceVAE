from argparse import ArgumentParser
import json
import shutil

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
from svae.utils.training import save_checkpoint, Params


if __name__ == '__main__':
    parser = ArgumentParser(description="Training of Sentence VAE")
    parser.add_argument("--config", type=str, required=True, metavar='PATH',
                        help="Path to a configuration file.")
    parser.add_argument("--run-dir", type=str, required=True, metavar='PATH',
                        help="Path to a directory where model checkpoints will be stored.")
    parser.add_argument("--force", action='store_true',
                        help="Whether to rewrite data if run directory already exists.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.exists() and args.force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    writer = SummaryWriter(logdir=str(run_dir))

    shutil.copyfile(args.config, run_dir / f"config.jsonnet")
    config = json.loads(evaluate_file(args.config))
    params = Params(config)
    training_params = params.pop('training')
    dataset_params = params.pop('dataset')

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
                                                            max_len=90)
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

    iters = 0
    for epoch in range(training_params.epochs):
        print("#"*20)
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
        # Validation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dev_iter, desc='Validation'):
                output = model(batch)
            metrics = model.get_metrics(reset=True)
            for metric, value in metrics.items():
                writer.add_scalar(f'dev/{metric}', value, epoch)

        for temperature in [0.1, 1., 10.]:
            print("#" * 20)
            print(f"Sentence samples. Temperature: {temperature}")
            samples = model.sample(num_samples=10, temperature=temperature, device=device)
            print(*samples, sep='\n')

    with (run_dir / 'TEXT.Field').open("wb") as fp:
        dill.dump(TEXT, fp)
    save_checkpoint(model.state_dict(), run_dir)

    if params.get('eval_on_test', False):
        print("Evaluating model on test data...")
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_iter, desc='Test set evaluation'):
                output = model(batch)
            metrics = model.get_metrics(reset=True)
            for metric, value in metrics.items():
                print(f"{metric}: {value}")

    writer.close()
