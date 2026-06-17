import os
import sys
import random
import warnings
from argparse import ArgumentParser
from datetime import timedelta

import numpy as np
import torch
import yaml
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy


class _ToChannelsLast:
    def __call__(self, x):
        return x.to(memory_format=torch.channels_last)


torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set as {seed}")


def get_args():
    parser = ArgumentParser(description='ConText-CIR Training')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--approx_steps', type=int, default=35000)
    parser.add_argument('--cross_layers', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=500)

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--lambda_cc', type=float, default=0.08)
    parser.add_argument('--epsilon_cc', type=float, default=0.0)
    parser.add_argument('--max_nps', type=int, default=10)

    parser.add_argument('--lr_schedule', type=str, default='cosine_restarts',
                        choices=['cosine_restarts', 'cosine'])
    parser.add_argument('--cosine_t0', type=int, default=2500)
    parser.add_argument('--backbone_lr_scale', type=float, default=1.0,
                        help='LR multiplier for the CLIP backbone + visual_projection relative to '
                             'the fresh fusion/pooler/query_projection. 1.0 = uniform.')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--grad_checkpoint', action='store_true',
                        help='Activation checkpointing on the CLIP encoders (memory for compute).')

    parser.add_argument('--strategy', type=str, default='ddp')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='bf16-mixed',
                        choices=['32', '16-mixed', 'bf16-mixed'])

    parser.add_argument('--project_name', type=str, default='ConText-CIR')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--wandb_mode', type=str, default='offline')

    parser.add_argument('--backbone_size', type=str, default='B16', choices=['B', 'B16', 'L', 'H'])

    parser.add_argument('--reload', action='store_true', help='Resume from last/best checkpoint')
    parser.add_argument('--init_weights', type=str, default=None,
                        help='Load model weights only (state_dict, strict=False) before training.')
    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--find_unused_parameters', action='store_true')
    parser.add_argument('--use_channel_last', action='store_true')

    parser.add_argument('--nps_cache', type=str, default='',
                        help='Precomputed NP SQLite cache (nps_precompute.py); used when lambda_cc>0. '
                             'Overrides config np_cache.')
    parser.add_argument('--bench_every_n_steps', type=int, default=2000)
    parser.add_argument('--bench_batch_size', type=int, default=256)
    parser.add_argument('--bench_num_workers', type=int, default=16)
    parser.add_argument('--bench_warmup', type=int, default=0)
    parser.add_argument('--no_circo', action='store_true', help='Skip CIRCO eval (CIRR only).')
    return parser.parse_args()


def create_transforms(use_channel_last=False):
    t = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    if use_channel_last:
        t.append(_ToChannelsLast())
    t.append(transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]))
    return transforms.Compose(t)


# kept for callers that import the old name
create_optimized_transforms = create_transforms


def setup_ddp_strategy(args):
    if args.devices > 1:
        return DDPStrategy(
            find_unused_parameters=args.find_unused_parameters,
            static_graph=not args.find_unused_parameters and args.lambda_cc == 0,
            gradient_as_bucket_view=True,
            timeout=timedelta(hours=2),  # rank-0 gallery encode can exceed the 30-min NCCL default
        )
    return 'auto'


if __name__ == '__main__':
    set_seed(0)
    args = get_args()
    print('Starting ConText-CIR Training')
    print(args)

    cfg = yaml.safe_load(open('config.yaml', 'r'))
    out_dir = args.output_dir or os.path.join(cfg['output'], args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    lr_logger = LearningRateMonitor(logging_interval='step')
    wb_logger = WandbLogger(save_dir=out_dir, project=args.project_name, name=args.run_name,
                            mode=args.wandb_mode, reinit=True)

    preproc = create_transforms(use_channel_last=args.use_channel_last)

    from Models import ConText
    from cir_dataset import CIRTripletDataset, collate_fn_with_nps

    model = ConText(
        args=args, num_cross_attn_layers=args.cross_layers, heads=8, dropout=0.0,
        weight_decay=args.weight_decay, temperature=args.temperature,
        steps_per_epoch=1000, max_epochs=50, warmup_steps=args.warmup_steps,
        lambda_cc=args.lambda_cc, epsilon_cc=args.epsilon_cc, max_nps=args.max_nps,
    )

    if args.init_weights:
        print(f"[init_weights] loading model weights from {args.init_weights}")
        sd = torch.load(args.init_weights, map_location='cpu', weights_only=False)
        sd = sd.get('state_dict', sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[init_weights] missing={len(missing)} unexpected={len(unexpected)}")

    if args.use_channel_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    # Training data: a JSON manifest of {"reference","target","caption"} triplets. NPs are attached
    # (for the Text-CC loss) only when lambda_cc>0, from the SQLite cache + spaCy fallback.
    cir_parser = model.parser if args.lambda_cc > 0 else None
    np_cache_path = args.nps_cache or cfg.get('np_cache') or None
    train_dataset = CIRTripletDataset(
        manifest_path=cfg['train_manifest'], image_root=cfg['image_root'], preprocess=preproc,
        parser=cir_parser, max_nps=args.max_nps, np_cache_path=np_cache_path,
    )

    num_workers = args.num_workers if args.num_workers > 0 else min(16, os.cpu_count() // 2)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=args.accelerator == 'gpu' and torch.cuda.is_available(),
        persistent_workers=num_workers > 0, prefetch_factor=6 if num_workers > 0 else None,
        collate_fn=collate_fn_with_nps, drop_last=True,
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )

    epochs = args.approx_steps // len(train_loader) + 1
    model.steps_per_epoch = len(train_loader)
    model.max_epochs = epochs
    print(f"Steps per epoch: {len(train_loader)} | approx. epochs: {epochs}")

    from cir_benchmarks import CIRBenchmarkCallback
    bench_cb = CIRBenchmarkCallback(
        cirr_path=cfg['cirr_eval_path'], circo_path=cfg['circo_eval_path'], preprocess=preproc,
        out_dir=out_dir, every_n_steps=args.bench_every_n_steps, batch_size=args.bench_batch_size,
        num_workers=args.bench_num_workers, monitor='bench/score',
        run_circo=not args.no_circo, warmup_until=args.bench_warmup,
    )
    ckpt_monitor = ModelCheckpoint(dirpath=out_dir, save_last=True, save_top_k=0,
                                   every_n_train_steps=args.bench_every_n_steps)
    callbacks = [bench_cb, lr_logger, ckpt_monitor]

    trainer = pl.Trainer(
        max_steps=args.approx_steps, max_epochs=-1, num_sanity_val_steps=0,
        strategy=setup_ddp_strategy(args), logger=wb_logger, accelerator=args.accelerator,
        devices=args.devices, callbacks=callbacks, log_every_n_steps=15,
        accumulate_grad_batches=args.accumulate_grad_batches, gradient_clip_val=args.gradient_clip_val,
        precision=args.precision, benchmark=True, sync_batchnorm=args.devices > 1,
    )

    ckpt_path = None
    if args.reload:
        for cand in ('last.ckpt', 'best.ckpt'):
            p = os.path.join(out_dir, cand)
            if os.path.exists(p):
                ckpt_path = p
                break
        print(f"Reloading from {ckpt_path}" if ckpt_path else "No checkpoint to reload; starting fresh")

    fit_kwargs = dict(train_dataloaders=train_loader)
    if ckpt_path is not None:
        fit_kwargs['ckpt_path'] = ckpt_path
    trainer.fit(model, **fit_kwargs)

    print(f"Training completed. Checkpoints under: {out_dir}")
