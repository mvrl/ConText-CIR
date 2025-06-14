# ==================== Train.py ====================
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
import torch
import os
import random
from argparse import ArgumentParser
import sys
import warnings
import torchvision.transforms as transforms
import yaml
from contextlib import nullcontext
import torch._dynamo
from torch.cuda.amp import GradScaler, autocast

# Set PyTorch optimizations
torch.set_float32_matmul_precision('high')
torch._dynamo.config.suppress_errors = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: deterministic mode is disabled for performance
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
    parser.add_argument('--strategy', type=str, default='ddp')
    
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['32', '16-mixed', 'bf16-mixed'])
    
    parser.add_argument('--project_name', type=str, default='ConText-CIR')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--wandb_mode', type=str, default='offline')
    
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--backbone_size', type=str, default='B', choices=['B', 'L', 'H'])
    
    parser.add_argument('--reload', action='store_true', help='Reload model from checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to save output logs and checkpoints')
    
    # Performance optimizations
    parser.add_argument('--compile_model', action='store_true', help='Use torch.compile for faster training')
    parser.add_argument('--compile_mode', type=str, default='default', choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument('--find_unused_parameters', action='store_true', help='Find unused parameters in DDP')
    parser.add_argument('--use_channel_last', action='store_true', help='Use channels_last memory format for better performance')
    
    args = parser.parse_args()
    return args


class OptimizedDataLoader:
    """Wrapper for DataLoader with performance optimizations."""
    
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory, collate_fn, 
                 persistent_workers=True, prefetch_factor=2, drop_last=False):
        
        # Use larger prefetch factor for better GPU utilization
        prefetch_factor = min(prefetch_factor * 2, 8) if num_workers > 0 else None
        
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn,
            drop_last=drop_last,
            multiprocessing_context='spawn' if num_workers > 0 else None,
        )
    
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        return iter(self.loader)


class CompiledConText(pl.LightningModule):
    """Wrapper to handle torch.compile with PyTorch Lightning."""
    
    def __init__(self, model, compile_model=False, compile_mode='default'):
        super().__init__()
        self.model = model
        
        # Copy attributes from original model
        for attr in ['hparams', 'learning_rate', 'temperature', 'lambda_cc', 
                     'epsilon_cc', 'steps_per_epoch', 'max_epochs']:
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))
        
        # Compile model if requested
        if compile_model and torch.cuda.is_available():
            print(f"Compiling model with mode: {compile_mode}")
            self.model = torch.compile(self.model, mode=compile_mode)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    
    def on_train_epoch_start(self):
        if hasattr(self.model, 'on_train_epoch_start'):
            self.model.on_train_epoch_start()


def create_optimized_transforms(use_channel_last=False):
    """Create transforms with optional channel_last format."""
    base_transforms = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    
    if use_channel_last:
        base_transforms.append(transforms.Lambda(lambda x: x.to(memory_format=torch.channels_last)))
    
    base_transforms.append(
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    )
    
    return transforms.Compose(base_transforms)


def setup_ddp_strategy(args):
    """Setup optimized DDP strategy."""
    if args.devices > 1:
        return DDPStrategy(
            find_unused_parameters=args.find_unused_parameters,
            static_graph=not args.find_unused_parameters,  # Enable static graph for better performance
            gradient_as_bucket_view=True,  # Optimize gradient communication
        )
    return 'auto'


if __name__ == '__main__':
    # Main training script
    set_seed(0)
    args = get_args()
    
    print('Starting ConText-CIR Training with Optimizations')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
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
    
    # Create optimized preprocessing
    preproc = create_optimized_transforms(use_channel_last=args.use_channel_last)
    
    # Import models and dataset
    from Models import ConText
    from Dataset import CIRDataset, collate_fn_with_nps
    
    # Initialize model with gradient checkpointing if requested
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
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        # Enable for text encoder
        if hasattr(model.text_encoder.model, 'gradient_checkpointing_enable'):
            model.text_encoder.model.gradient_checkpointing_enable()
        # Enable for vision encoder
        if hasattr(model.vision_encoder.model, 'gradient_checkpointing_enable'):
            model.vision_encoder.model.gradient_checkpointing_enable()
    
    # Convert to channels_last if requested
    if args.use_channel_last and torch.cuda.is_available():
        print("Converting model to channels_last memory format...")
        model = model.to(memory_format=torch.channels_last)
    
    # Get parser from model
    parser = model.parser
    
    # Load datasets with parser
    dataset_names = args.datasets.lower().split(',')
    train_datasets = []
    val_datasets = []
    
    for dataset in dataset_names:
        try:
            if dataset == 'cirr':
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
            elif dataset == 'cirr_r':
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
            elif dataset == 'hotels':
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
            elif dataset == 'lasco':
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
            else:
                raise ValueError(f"Dataset {dataset} not found")
        except Exception as e:
            raise Exception(f"Error loading dataset {dataset}: {e}")
    
    # Combine datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    # Optimize DataLoader settings
    num_workers = args.num_workers if args.num_workers > 0 else min(8, os.cpu_count() // 2)
    
    # Create optimized dataloaders
    train_loader = OptimizedDataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.accelerator == 'gpu' and torch.cuda.is_available(),
        collate_fn=collate_fn_with_nps,
        drop_last=True,  # Drop last for consistent batch sizes
    ).loader
    
    val_loader = OptimizedDataLoader(
        val_dataset, 
        batch_size=args.val_batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.accelerator == 'gpu' and torch.cuda.is_available(),
        collate_fn=collate_fn_with_nps,
        drop_last=False,
    ).loader
    
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
    
    # Wrap model with compile wrapper if requested
    if args.compile_model:
        model = CompiledConText(model, compile_model=True, compile_mode=args.compile_mode)
    
    # Setup callbacks
    callbacks = [ckpt_monitor, lr_logger, early_stop_callback]
    
    # Add profiler for performance debugging (optional)
    if os.environ.get('PROFILE_TRAINING', '0') == '1':
        from pytorch_lightning.profilers import PyTorchProfiler
        profiler = PyTorchProfiler(
            dirpath=out_dir,
            filename='perf_logs',
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(out_dir, 'tb_logs')),
            record_shapes=True,
            with_stack=True,
        )
    else:
        profiler = None
    
    # Initialize Trainer with optimizations
    trainer = pl.Trainer(
        strategy=setup_ddp_strategy(args),
        max_epochs=epochs, 
        logger=wb_logger, 
        accelerator=args.accelerator, 
        devices=args.devices, 
        callbacks=callbacks, 
        log_every_n_steps=15,
        num_sanity_val_steps=1,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        precision=args.precision,
        profiler=profiler,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        benchmark=True,  # Enable cudnn.benchmark
        sync_batchnorm=args.devices > 1,  # Sync batchnorm for multi-GPU
    )
    
    # Train the model
    if args.reload:
        models = [path for path in os.listdir(out_dir) if path.endswith('.ckpt')]
        models = [m for m in models if 'last' not in m]
        if len(models) > 0:
            val_losses = []
            for m in models:
                try:
                    val_loss = float(m.split('val_loss=')[-1].strip('.ckpt').split('-')[0])
                    val_losses.append(val_loss)
                except:
                    val_losses.append(float('inf'))
            
            best_model = models[np.argmin(val_losses)]
            ckpt_path = os.path.join(out_dir, best_model)
            print(f"Reloading model from {ckpt_path}")
            
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        else:
            print("No model to reload, starting fresh training")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("Training completed!")
    print(f"Best model saved at: {ckpt_monitor.best_model_path}")