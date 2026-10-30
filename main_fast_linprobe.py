# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

"""Single-process linear probing with cached, frozen encoder features.

Fast linprobe uses no data augmentation: each sample produces one fixed feature
vector, reused for every classifier epoch. Compared with regular linprobe using
fresh augmented views, accuracy is usually slightly lower, but evaluation is
substantially faster because the encoder only runs during feature extraction.
"""

import argparse
import datetime
import importlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
import util.yaml_loader as config_util
from engine_fast_linprobe import evaluate, extract_features, train_one_epoch
from model_amr import LinearProbeHead
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def import_class(name):
    module_path, _, cls_name = name.rpartition('.')
    return getattr(importlib.import_module(module_path), cls_name)


def get_args_parser():
    parser = argparse.ArgumentParser('AMR fast linear probing', add_help=False)
    parser.add_argument('--config', default='./config/linprobe.yaml',
                        help='path to the configuration file')
    parser.add_argument('--cfg-name', default='ntu60_xsub',
                        help='named config entry within --config')

    parser.add_argument('--batch_size', default=512, type=int,
                        help='Classifier batch size (effective batch size is batch_size * accum_iter)')
    parser.add_argument('--feature_batch_size', default=128, type=int,
                        help='Batch size for the single encoder pass over each split')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Number of classifier batches per optimizer step')

    # Model parameters
    parser.add_argument('--model', default='model_amr.DownstreamTransformer', type=str,
                        help='Frozen feature extractor class')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument('--finetune', default='', help='pretrained encoder checkpoint')

    # Optimizer parameters, matching the regular linear-probe entry point.
    parser.add_argument('--enable_amp', action='store_true', default=False,
                        help='Enable mixed precision for classifier training on CUDA; encoder features stay float32')
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--lr', default=None, type=float, help='absolute learning rate')
    parser.add_argument('--blr', default=0.1, type=float,
                        help='base learning rate: lr = blr * effective_batch_size / 256')
    parser.add_argument('--min_lr', default=0., type=float)
    parser.add_argument('--warmup_epochs', default=10, type=int)

    # Dataset parameters. Random augmentations are disabled for both splits.
    parser.add_argument('--dataset', default='util.datasets.SkeletonDataset')
    parser.add_argument('--train_dataset_args', default=dict())
    parser.add_argument('--val_dataset_args', default=dict())
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Data loader workers for feature extraction')
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--output_dir', default='./output/fast_linprobe',
                        help='Checkpoint and JSON log directory; empty to disable both')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the final epoch model in output_dir (default: no model saving)')
    parser.add_argument('--log_dir', default='./output/fast_linprobe',
                        help='TensorBoard log directory; empty to disable logging')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='Load classifier weights; --finetune still supplies the frozen encoder')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the classifier supplied by --resume')
    return parser


def parse_args(argv=None):
    parser = argparse.ArgumentParser('AMR fast linear probing', parents=[get_args_parser()])
    config_args, _ = parser.parse_known_args(argv)
    if config_args.config:
        defaults = config_util.load_yaml_config(config_args.config, cfg_name=config_args.cfg_name)
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def without_augmentation(dataset_args):
    """Keep deterministic resizing/modalities while removing random augmented views."""
    dataset_args = dict(dataset_args)
    for name in ('random_choose', 'random_shift', 'random_move', 'random_rot'):
        dataset_args[name] = False
    dataset_args['p_interval'] = [1.0]
    return dataset_args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_pretrained_encoder(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    model_keys = model.state_dict().keys()
    encoder_weights = {}
    for name, value in checkpoint_model.items():
        while name.startswith(('module.', '_orig_mod.')):
            name = name.split('.', 1)[1]
        if name.startswith('head.') or name not in model_keys:
            continue
        if name in encoder_weights:
            raise ValueError(f'Duplicate encoder parameter after removing wrappers: {name}')
        encoder_weights[name] = value

    msg = model.load_state_dict(encoder_weights, strict=False)
    missing_encoder = [name for name in msg.missing_keys if not name.startswith('head.')]
    if missing_encoder:
        raise ValueError(f'Pretrained checkpoint is missing encoder weights: {missing_encoder}')
    print(f'Load pre-trained checkpoint from: {checkpoint_path}')
    print(msg)


def main(args):
    if not args.finetune:
        raise ValueError('--finetune must specify a pretrained encoder checkpoint')
    if args.eval and not args.resume:
        raise ValueError('--eval requires classifier weights supplied by --resume')
    if args.batch_size < 2 or args.feature_batch_size < 1 or args.accum_iter < 1:
        raise ValueError('batch_size must be at least 2; feature_batch_size and accum_iter must be positive')
    if args.epochs < 1 and not args.eval:
        raise ValueError('epochs must be positive for classifier training')

    device = torch.device(args.device)
    args.enable_amp = args.enable_amp and device.type == 'cuda'
    args.train_dataset_args = without_augmentation(args.train_dataset_args)
    args.val_dataset_args = without_augmentation(args.val_dataset_args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))
    print('Fast linprobe: no data augmentation; fixed features trade a little accuracy for much faster training.')

    # Fix all random seeds, including the classifier initialization and shuffle.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    Model = import_class(args.model)
    encoder = Model(**args.model_args)
    encoder.return_features = True
    load_pretrained_encoder(encoder, args.finetune)
    encoder.requires_grad_(False)
    encoder.to(device)
    num_classes = encoder.num_classes

    # Match linear_eval.py: keep nn.Linear's default initialization and create
    # the classifier before feature-loader iteration consumes random seeds.
    # Pooling flattens the encoded joints (num_joints // patch_size) and channels.
    feature_dim = encoder.dim_feat * (encoder.num_joints // encoder.patch_size)
    model = LinearProbeHead(dim_feat=feature_dim, num_classes=num_classes).to(device)

    # Stage 1: one deterministic encoder pass per sample. Cache features on CPU
    # so GPU memory use does not grow with the size of the dataset.
    Dataset = import_class(args.dataset)
    feature_datasets = {}
    extraction_start = time.time()
    splits = [('val', args.val_dataset_args)] if args.eval else [
        ('train', args.train_dataset_args), ('val', args.val_dataset_args)]
    for split, dataset_args in splits:
        dataset = Dataset(**dataset_args)
        print(dataset)
        loader = DataLoader(dataset, sampler=SequentialSampler(dataset),
                            batch_size=args.feature_batch_size, num_workers=args.num_workers,
                            worker_init_fn=seed_worker, pin_memory=args.pin_mem, drop_last=False)
        # The original linear_eval extracts features in float32 even when AMP
        # is enabled for classifier training; preserve that evaluation protocol.
        features, labels = extract_features(encoder, loader, device, split=split)
        feature_datasets[split] = TensorDataset(features, labels)
        del dataset, loader, features, labels
    print('Feature extraction time {}'.format(
        datetime.timedelta(seconds=int(time.time() - extraction_start))))
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Stage 2: only the standalone BN + linear classifier is trained.
    val_features = feature_datasets['val'].tensors[0]
    if model.net[1].in_features != val_features.shape[1]:
        raise ValueError('Classifier width does not match the extracted encoder features')
    n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print('Model = %s' % model)
    print('number of trainable params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print('base lr: %.2e' % (args.lr * 256 / eff_batch_size))
    print('actual lr: %.2e' % args.lr)
    print('accumulate grad iterations: %d' % args.accum_iter)
    print('effective batch size: %d' % eff_batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler(enabled=args.enable_amp)
    criterion = torch.nn.CrossEntropyLoss()
    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    # Cached tensors need no worker processes or further data augmentation.
    data_loader_val = DataLoader(feature_datasets['val'], sampler=SequentialSampler(feature_datasets['val']),
                                batch_size=args.batch_size, num_workers=0, pin_memory=args.pin_mem)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy on {len(feature_datasets['val'])} validation samples: {test_stats['acc1']:.1f}%")
        return test_stats

    cached_train = feature_datasets['train']
    if len(cached_train) < 2:
        raise ValueError('At least two training samples are needed for BatchNorm')
    if cached_train.tensors[0].shape[1] != val_features.shape[1]:
        raise ValueError('Training and validation feature widths do not match')
    data_loader_train = DataLoader(
        cached_train, sampler=RandomSampler(cached_train), batch_size=args.batch_size,
        num_workers=0, pin_memory=args.pin_mem,
        # BatchNorm cannot train on a singleton final batch; retain other tails.
        drop_last=len(cached_train) % args.batch_size == 1)
    log_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    max_accuracy = 0.0
    try:
        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
                max_norm=None, log_writer=log_writer, args=args)
            if args.save_model and args.output_dir and epoch == args.epochs - 1:
                # Save only the classifier; --finetune supplies the frozen encoder.
                misc.save_model(args=args, epoch=epoch, model=model, model_without_ddp=model,
                                optimizer=optimizer, loss_scaler=loss_scaler)

            test_stats = evaluate(data_loader_val, model, device)
            max_accuracy = max(max_accuracy, test_stats['acc1'])
            print(f"Accuracy on {len(feature_datasets['val'])} validation samples: {test_stats['acc1']:.1f}%")
            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                for key in ('acc1', 'acc5', 'loss'):
                    log_writer.add_scalar(f'perf/test_{key}', test_stats[key], epoch)
                log_writer.flush()

            log_stats = {**{f'train_{key}': value for key, value in train_stats.items()},
                         **{f'test_{key}': value for key, value in test_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters}
            if args.output_dir:
                with open(Path(args.output_dir) / 'log.txt', 'a', encoding='utf-8') as log_file:
                    log_file.write(json.dumps(log_stats) + '\n')
    finally:
        if log_writer is not None:
            log_writer.close()

    print('Training time {}'.format(datetime.timedelta(seconds=int(time.time() - start_time))))
    return max_accuracy


if __name__ == '__main__':
    main(parse_args())
