# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import yaml
import numpy as np
import os
import time
from pathlib import Path

import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim as optim_factory

import util.misc as misc
import util.yaml_loader as config_util
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_pretrain import train_one_epoch

# torch.distributed.barrier(device_ids=int(os.environ["LOCAL_RANK"]))

def import_class(name):
    import importlib
    module_path, _, cls_name = name.rpartition('.')
    return getattr(importlib.import_module(module_path), cls_name)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--config', default='./config/ntu60_xsub_joint_pretrain_debug.yaml', help='path to the configuration file')
    parser.add_argument('--cfg-name', default=None, help='named config entry within --config file (e.g. ntu60_xsub)')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument('--decoder_attn_mode', default='sigmoid',
                        choices=['sigmoid', 'sigmoid_scaled', 'softmax'],
                        help='Override decoder cross-attention mode in model_args.')
    parser.add_argument('--decoder_attn_alpha', default=1.0, type=float,
                        help='Fixed positive softmax attention multiplier (alpha).')
    parser.add_argument('--decoder_attn_temperature', default=1.0, type=float,
                        help='Fixed positive softmax attention temperature (T).')

    parser.add_argument('--mask_ratio', default=0.90, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    parser.add_argument('--motion_stride', default=1, type=int,
                        help='stride of motion to be predicted.')
    parser.add_argument('--motion_aware_tau', default=0.75, type=float,
                        help='temperature of motion aware masking.')
    parser.add_argument('--fusion_weights', default=[0.34, 0.33, 0.33], type=float, nargs=3,
                        help='weights for the 3 window scales (4,8,12) in adaptive_weights.')
                        
    parser.add_argument('--mask_ratio_inter', default=0.75, type=float,
                        help='Masking ratio inter (percentage of removed patches).')
    parser.add_argument('--mask_ratio_intra', default=0.80, type=float,
                        help='Masking ratio intra (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--enable_amp', action='store_true', default=True,
                        help='Enabling automatic mixed precision')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--beta1', type=float, default=0.875,
                        help='AdamW beta1 (decay rate for the first moment).')
    parser.add_argument('--beta2', type=float, default=0.950,
                        help='AdamW beta2 (decay rate for the second moment).')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='util.datasets.SkeletonDataset', help='dataset class to use')
    parser.add_argument('--train_dataset_args', default=dict(), help='the arguments of dataset for training')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    
    misc.init_distributed_mode(args)

    # CLI overrides take precedence over the nested YAML model_args.
    args.model_args = dict(args.model_args)
    for name in ('decoder_attn_mode', 'decoder_attn_alpha', 'decoder_attn_temperature'):
        value = getattr(args, name, None)
        if value is not None:
            args.model_args[name] = value

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Load dataset
    SkeletonDataset = import_class(args.dataset)
    dataset_train = SkeletonDataset(**args.train_dataset_args)
    print(dataset_train)

    global_rank = misc.get_rank()

    if args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # if args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    def worker_init_fn(worker_id):                                                          
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    Model = import_class(args.model)
    model = Model(**args.model_args)



    model.to(device)

    # model = torch.compile(model)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
  
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    
    p = parser.parse_args()
    if p.config is not None:
        default_args = config_util.load_yaml_config(p.config, cfg_name=p.cfg_name)
        parser.set_defaults(**default_args)

    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
