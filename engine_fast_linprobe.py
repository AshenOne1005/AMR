# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the root directory of this source tree.

"""Feature extraction and classifier training for single-process fast linprobe."""

import math

import torch

import util.lr_sched as lr_sched
import util.misc as misc


@torch.no_grad()
def extract_features(model, data_loader, device, enable_amp=False, split='train'):
    """Extract each sample once and cache detached, float32 features on the CPU."""
    if len(data_loader) == 0:
        raise ValueError(f'The {split} feature loader is empty')
    model.eval()
    metric_logger = misc.MetricLogger(delimiter='  ')
    features, labels = [], []
    for batch in metric_logger.log_every(data_loader, 20, f'Extract {split}:'):
        samples, targets = batch[:2]
        samples = samples.float().to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=enable_amp):
            outputs = model(samples)
        if outputs.ndim != 2:
            raise ValueError(f'Expected pooled features shaped (N, D), got {outputs.shape}')
        features.append(outputs.float().cpu())
        labels.append(targets.long().cpu())

    # Concatenate once instead of repeatedly copying all accumulated features.
    features = torch.cat(features)
    labels = torch.cat(labels)
    print(f'{split} features: {tuple(features.shape)}')
    return features, labels


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    loss_scaler, max_norm=None, log_writer=None, args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    accum_iter = args.accum_iter
    num_batches = len(data_loader)
    if num_batches == 0:
        raise ValueError('The classifier training loader is empty')
    optimizer.zero_grad()

    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 50, header)):
        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, step / num_batches + epoch, args)

        samples = samples.float().to(device, non_blocking=True)
        targets = targets.long().to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=args.enable_amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f'Loss is {loss_value}, stopping training')

        # Also step the optimizer for the final, possibly incomplete window.
        window_start = (step // accum_iter) * accum_iter
        window_size = min(accum_iter, num_batches - window_start)
        update_grad = (step + 1) % accum_iter == 0 or step + 1 == num_batches
        loss_scaler(loss / window_size, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), update_grad=update_grad)
        if update_grad:
            optimizer.zero_grad()

        lr = max(group['lr'] for group in optimizer.param_groups)
        metric_logger.meters['loss'].update(loss_value, n=samples.shape[0])
        metric_logger.update(lr=lr)
        if log_writer is not None and update_grad:
            epoch_1000x = int((step / num_batches + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    print('Averaged stats:', metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    if len(data_loader) == 0:
        raise ValueError('The classifier validation loader is empty')
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter='  ')

    for samples, targets in metric_logger.log_every(data_loader, 50, 'Test:'):
        samples = samples.float().to(device, non_blocking=True)
        targets = targets.long().to(device, non_blocking=True)
        outputs = model(samples)
        loss = criterion(outputs, targets)
        topk = min(5, outputs.shape[1])
        predictions = outputs.topk(topk, dim=1).indices
        correct = predictions.eq(targets[:, None])
        acc1 = correct[:, :1].any(dim=1).float().mean().item() * 100
        acc5 = correct.any(dim=1).float().mean().item() * 100
        batch_size = samples.shape[0]
        for key, value in (('loss', loss.item()), ('acc1', acc1), ('acc5', acc5)):
            metric_logger.meters[key].update(value, n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}
