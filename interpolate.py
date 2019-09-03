from argparse import ArgumentParser
import dill
import json
from pathlib import Path

from _jsonnet import evaluate_file
import numpy as np

import torch
import torch.nn as nn
from torchtext.data import Field

from svae.svae import RecurrentVAE
from svae.utils.training import Params
from svae.dataset_utils import UNK_TOKEN
from svae.utils.interpolation import slerp, lerp


def encode_sentence(model: nn.Module, sentence: str,
                    field: Field, device: torch.device) -> torch.Tensor:
    tokens = field.tokenize(sentence)
    vocab = field.vocab
    sentence_indices = [vocab.stoi[token] if token in vocab.stoi else vocab.stoi[UNK_TOKEN]
                        for token in tokens]
    sentence_numerical = torch.LongTensor([sentence_indices])
    sentence_numerical = sentence_numerical.to(device)
    # batch_len x seq_len
    sentence_numerical = sentence_numerical.view(-1, 1)
    with torch.no_grad():
        z = model.encode(sentence_numerical, torch.LongTensor([sentence_numerical.size(1)]), use_mean=True)['code']
    return z


if __name__ == '__main__':
    parser = ArgumentParser(description="Interpolation between points in latent space")
    parser.add_argument("--model-path", type=str, required=True, metavar="PATH",
                        help="Path to model data")
    parser.add_argument("--start-sentence", type=str, required=False, metavar="TEXT",
                        help="Start sentence for interpolation")
    parser.add_argument("--end-sentence", type=str, required=False, metavar="TEXT",
                        help="End sentence for interpolation")
    parser.add_argument("--num-steps", type=int, default=8, metavar="N",
                        help="Number of interpolation steps (default: 10)")
    parser.add_argument("--interpolation-type", type=str, default='linear', metavar="TYPE",
                        choices=['linear', 'spherical'], help="Interpolation type")
    args = parser.parse_args()

    # Load model
    model_dir = Path(args.model_path)
    config = json.loads(evaluate_file(str(model_dir / 'config.jsonnet')))
    params = Params(config)

    with (model_dir / 'TEXT.Field').open("rb") as fp:
        TEXT: Field = dill.load(fp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecurrentVAE(vocab=TEXT.vocab, params=params.pop('model'))
    model.load_state_dict(torch.load(model_dir / 'vae.pt'))
    model.greedy = True
    model.to(device)
    model.eval()

    # Prepare data
    if args.start_sentence is None or args.end_sentence is None:
        z_1 = np.random.randn(1, model.latent_dim)
        z_2 = np.random.randn(1, model.latent_dim)
    else:
        print(f"Original sentences:\n"
              f"Start sentence: {args.start_sentence}\n"
              f"End sentence: {args.end_sentence}.")
        z_1 = encode_sentence(model, args.start_sentence, TEXT, device)
        z_2 = encode_sentence(model, args.end_sentence, TEXT, device)
        z_1, z_2 = z_1.cpu().data.numpy(), z_2.cpu().data.numpy()

    z_steps = slerp(z_1, z_2, num_steps=args.num_steps)
    codes = torch.FloatTensor(z_steps)
    samples = model.sample(z=codes, device=device)
    print("\n".join(samples))
