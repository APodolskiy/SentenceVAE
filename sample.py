from argparse import ArgumentParser
import dill
import json
from pathlib import Path

from _jsonnet import evaluate_file
import torch
from torchtext.data import Field

from svae.svae import RecurrentVAE
from svae.utils.training import Params


if __name__ == '__main__':
    parser = ArgumentParser(description="Script for sampling utterances from models.")
    parser.add_argument("--model-path", type=str, required=True, metavar='PATH',
                        help="Path to model data.")
    parser.add_argument("--num-samples", type=int, default=10, metavar='N',
                        help="Number of samples to generate.")
    args = parser.parse_args()

    # Load model
    model_dir = Path(args.model_path)
    config = json.loads(evaluate_file(str(model_dir / 'config.jsonnet')))
    params = Params(config)

    with (model_dir / 'TEXT.Field').open("rb") as fp:
        TEXT: Field = dill.load(fp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model: RecurrentVAE = RecurrentVAE(vocab=TEXT.vocab, params=params.pop('model'))
    model.load_state_dict(torch.load(model_dir / 'vae.pt'))
    model.to(device)
    model.eval()

    # TODO: add different samplers
    samples = model.sample()
    print("\n".join(samples))
