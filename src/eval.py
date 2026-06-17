
import argparse

import torch
import yaml
from torchvision import transforms

from Models import ConText
from cir_benchmarks import evaluate_cirr, evaluate_circo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PREP = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to a ConText .ckpt')
    parser.add_argument('--benchmarks', nargs='+', default=['cirr', 'circo'],
                        choices=['cirr', 'circo'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    cfg = yaml.safe_load(open('config.yaml', 'r'))
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else None

    print(f'EVALUATING MODEL FROM {args.path}', flush=True)
    model = ConText.load_from_checkpoint(args.path).eval().to(device)

    metrics = {}
    if 'cirr' in args.benchmarks:
        metrics.update(evaluate_cirr(model, cfg['cirr_eval_path'], PREP, device,
                                     amp_dtype, args.batch_size, args.num_workers))
    if 'circo' in args.benchmarks:
        metrics.update(evaluate_circo(model, cfg['circo_eval_path'], PREP, device,
                                      amp_dtype, args.batch_size, args.num_workers))

    print('\n===== results =====')
    for k, v in sorted(metrics.items()):
        print(f'{k}: {v:.4f}')


if __name__ == '__main__':
    main()
