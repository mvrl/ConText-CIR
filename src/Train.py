# ==================== Train.py ====================
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import numpy as np
import torch
import os
import random
from argparse import ArgumentParser
import sys
import warnings
import torchvision.transforms as transforms
import yaml

torch.set_float32_matmul_precision('high')

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "300"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")


def get_args():
    parser = ArgumentParser(description='ConText-CIR Training')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--approx_steps', type=int, default=35000)
    parser.add_argument('--cross_layers', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--val_check_interval', type=int, default=500)
    
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--lambda_cc', type=float, default=0.08)
    parser.add_argument('--epsilon_cc', type=float, default=0.05)
    parser.add_argument('--max_nps', type=int, default=10)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_true')
    
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    
    parser.add_argument('--project_name', type=str, default='ConText-CIR')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--wandb_mode', type=str, default='offline')
    
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--backbone_size', type=str, default='B', choices=['B', 'L', 'H'])
    
    parser.add_argument('--reload', action='store_true', help='Reload model from checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to save output logs and checkpoints')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Main training script
    set_seed(0)
    args = get_args()
    
    print('Starting ConText-CIR Training')
    print(args)
    
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    
    # Initialize output directory
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(cfg['model_dir'], f'{args.datasets}/{args.backbone_size}/{args.run_name}')
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='step')
    wb_logger = WandbLogger(
        save_dir=out_dir,
        project=args.project_name, 
        name=args.run_name, 
        mode=args.wandb_mode,
        reinit=True
    )
    
    ckpt_monitor = ModelCheckpoint(
        monitor='val_loss', 
        filename='{epoch}-{step}-{val_loss:.3e}', 
        save_top_k=5, 
        save_last=True, 
        save_on_train_epoch_end=True,
        dirpath=out_dir
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=6,
        verbose=True,
        mode='min'
    )
    
    # Define CLIP preprocessing transform
    preproc = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
    ])
    
    # Import models and dataset
    from Models import ConText
    from Dataset import CIRDataset, collate_fn_with_nps
    
    # Initialize model first to get parser
    model = ConText(
        args=args,
        num_cross_attn_layers=args.cross_layers, 
        heads=8, 
        dropout=0.0,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        steps_per_epoch=1000,  # Will update later
        max_epochs=50,  # Will update later
        warmup_steps=args.warmup_steps,
        lambda_cc=args.lambda_cc,
        epsilon_cc=args.epsilon_cc,
        max_nps=args.max_nps
    )
    
    # Get parser from model
    parser = model.parser
    
    # Load datasets with parser
    dataset_names = args.datasets.lower().split(',')
    train_datasets = []
    val_datasets = []
    
    for dataset in dataset_names:
        try:
            match dataset:
                case 'cirr':
                    train_datasets.append(CIRDataset(
                        data_path=cfg['cirr_data_path'], 
                        split='train', 
                        dataset='cirr', 
                        preprocess=preproc,
                        parser=parser,
                        max_nps=args.max_nps
                    ))
                    val_datasets.append(CIRDataset(
                        data_path=cfg['cirr_data_path'], 
                        split='val', 
                        dataset='cirr', 
                        preprocess=preproc,
                        parser=parser,
                        max_nps=args.max_nps
                    ))
                case 'cirr_r':
                    train_datasets.append(CIRDataset(
                        data_path=cfg['cirr_data_path'], 
                        split='train', 
                        dataset='cirr_r', 
                        preprocess=preproc,
                        parser=parser,
                        max_nps=args.max_nps
                    ))
                    val_datasets.append(CIRDataset(
                        data_path=cfg['cirr_data_path'], 
                        split='val', 
                        dataset='cirr_r', 
                        preprocess=preproc,
                        parser=parser,
                        max_nps=args.max_nps
                    ))
                case 'hotels':
                    train_datasets.append(CIRDataset(
                        data_path=cfg['hotel_data_path'], 
                        split='train', 
                        dataset='hotels', 
                        preprocess=preproc,
                        parser=parser,
                        max_nps=args.max_nps
                    ))
                    val_datasets.append(CIRDataset(
                        data_path=cfg['hotel_data_path'], 
                        split='val', 
                        dataset='hotels', 
                        preprocess=preproc,
                        parser=parser,
                        max_nps=args.max_nps
                    ))
                case 'lasco':
                    train_datasets.append(CIRDataset(
                        data_path=cfg['lasco_data_path'], 
                        split='train', 
                        dataset='lasco', 
                        preprocess=preproc,
                        parser=parser,
                        max_nps=args.max_nps
                    ))
                    val_datasets.append(CIRDataset(
                        data_path=cfg['lasco_data_path'], 
                        split='val', 
                        dataset='lasco', 
                        preprocess=preproc,
                        parser=parser,
                        max_nps=args.max_nps
                    ))
                case _:
                    raise ValueError(f"Dataset {dataset} not found")
        except Exception as e:
            raise Exception(f"Error loading dataset {dataset}: {e}")
    
    # Combine datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    # Optimize DataLoader settings
    num_workers = args.num_workers if args.num_workers > 0 else os.cpu_count() // 2
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        num_workers=num_workers, 
        shuffle=True, 
        pin_memory=args.accelerator == 'gpu',
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn_with_nps
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.val_batch_size, 
        num_workers=num_workers, 
        shuffle=False,  
        pin_memory=args.accelerator == 'gpu',
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn_with_nps
    )
    
    epochs = args.approx_steps // len(train_loader) + 1
    print(f"#########################")
    print(f"Approx. epochs: {epochs}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"#########################")
    
    if args.val_check_interval > len(train_loader):
        args.val_check_interval = len(train_loader)
        print(f"Validation check interval set to {args.val_check_interval}")
    
    # Update model with correct steps_per_epoch and max_epochs
    model.steps_per_epoch = len(train_loader)
    model.max_epochs = epochs
    
    # Initialize Trainer
    trainer = pl.Trainer(
        strategy=args.strategy,
        max_epochs=epochs, 
        logger=wb_logger, 
        accelerator=args.accelerator, 
        devices=args.devices, 
        callbacks=[ckpt_monitor, lr_logger, early_stop_callback], 
        log_every_n_steps=15,
        num_sanity_val_steps=1,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
    )
    
    # Train the model
    if args.reload:
        models = [path for path in os.listdir(out_dir) if path.endswith('.ckpt')]
        models = [model for model in models if 'last' not in model]
        if len(models) > 0:
            val_losses = [float(model.split('=')[-1].strip('.ckpt').split('-v')[0]) for model in models]
            best_model = models[np.argmin(val_losses)]
            ckpt_path = os.path.join(out_dir, best_model)
            print(f"Reloading model from {ckpt_path}")
        else:
            print("No model to reload")
            raise ValueError("No model to reload")
        
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)