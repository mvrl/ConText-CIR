import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np
import torch
import os
import random
from argparse import ArgumentParser
import sys
import warnings
from .config import cfg

from .utils import *
from .Models import *
from .Dataset import *
from .Parser import *

# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

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
    parser = ArgumentParser(description='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--val_check_interval', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=30)
    
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
   
    parser.add_argument('--accelerator',type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    
    parser.add_argument('--project_name', type=str, default='ConText-CIR')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    
    # encoder types:
    parser.add_argument('--backbone_size', type=str, default='B', choices=['B', 'L', 'H'])

    
    # Training resuming parameters:
    parser.add_argument('--ckpt_path',type=str, default ='none')

    args = parser.parse_args()

    return args

def set_attrs(args):
    if args.devices > 1:
        setattr(args,'dist_train',True)
    else:
        setattr(args,'dist_train',False)
    
    return args


if __name__ == '__main__':
    set_seed(42)
    args = get_args()
    #set learning rate logger
    print('Starting Training')
    print(args)
    #initliaze model
    model = ConText(args)
    #initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='step')
    wb_logger = WandbLogger(save_dir=cfg.log_dir,project=args.project_name, name=args.run_name, mode=args.wandb_mode)
    ckpt_monitor1 = ((
            ModelCheckpoint(monitor='val_loss', filename='{epoch}-{step}-{val_loss:.3f}',save_top_k = 10, save_last=True,save_on_train_epoch_end=True)
        ))

    trainer = pl.Trainer(precision=16, max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy,
    accelerator=args.accelerator, devices=args.devices, callbacks=[ckpt_monitor1, lr_logger], 
    # val_check_interval=args.val_check_interval, 
    check_val_every_n_epoch=1, log_every_n_steps=15)
    
    if args.ckpt_path.lower()=='none':
        trainer.fit(model)
    else:
        trainer.fit(model, ckpt_path=args.ckpt_path)