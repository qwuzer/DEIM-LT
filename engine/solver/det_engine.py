"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        # Variable to store gradient norm for logging
        total_grad_norm = None
        
        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            elif writer and dist_utils.is_main_process() and global_step % 10 == 0:
                # Compute gradient norm for logging only when needed
                # Note: We compute from scaled gradients to avoid interfering with scaler state
                # This gives an approximation but is safe
                try:
                    # Compute norm from scaled gradients (will be scaled, but gives relative magnitude)
                    total_grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    # Note: This is scaled gradient norm, not true norm, but useful for monitoring
                except Exception:
                    # If anything fails, just skip gradient logging
                    total_grad_norm = None

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            elif writer and dist_utils.is_main_process() and global_step % 10 == 0:
                # Compute gradient norm for logging (before step clears gradients)
                try:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                except Exception:
                    total_grad_norm = None

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
            
            # Add gradient norms monitoring (safe with error handling)
            if total_grad_norm is not None:
                try:
                    writer.add_scalar('Gradients/total_norm', total_grad_norm, global_step)
                except Exception as e:
                    # Silently skip gradient logging if there's any error
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, writer=None, epoch=0):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    step_counter = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # Compute validation losses (with error handling)
        try:
            # Create metas dict similar to training
            global_step = epoch * len(data_loader) + step_counter
            metas = dict(epoch=epoch, step=step_counter, 
                        global_step=global_step, 
                        epoch_step=len(data_loader))
            loss_dict = criterion(outputs, targets, **metas)
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())
            
            # Update metric logger with losses
            metric_logger.update(loss=loss_value, **loss_dict_reduced)
        except Exception as e:
            # Silently skip loss computation if it fails (e.g., criterion doesn't support eval mode)
            # This ensures evaluation continues even if loss computation has issues
            pass
        
        step_counter += 1

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # Log validation losses to TensorBoard (if computed)
    if writer is not None and dist_utils.is_main_process():
        try:
            # Get loss stats from metric logger
            loss_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if 'loss' in k.lower()}
            if 'loss' in loss_stats:
                writer.add_scalar('Val/loss', loss_stats['loss'], epoch)
            # Log individual loss components
            for k, v in loss_stats.items():
                if k != 'loss':
                    writer.add_scalar(f'Val/{k}', v, epoch)
        except Exception:
            # Silently skip if loss logging fails
            pass
    
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # Include loss stats in return value
    try:
        stats.update({k: meter.global_avg for k, meter in metric_logger.meters.items()})
    except Exception:
        pass
    
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        
        # Log per-class mAP and COCO metrics breakdown (with comprehensive error handling)
        if writer is not None and dist_utils.is_main_process() and 'bbox' in iou_types:
            try:
                coco_eval = coco_evaluator.coco_eval['bbox']
                
                # Log detailed COCO metrics breakdown
                if 'coco_eval_bbox' in stats and len(stats['coco_eval_bbox']) >= 6:
                    # stats format: [AP, AP50, AP75, APs, APm, APl]
                    try:
                        writer.add_scalar('Val/mAP', stats['coco_eval_bbox'][0], epoch)
                        writer.add_scalar('Val/mAP50', stats['coco_eval_bbox'][1], epoch)
                        writer.add_scalar('Val/mAP75', stats['coco_eval_bbox'][2], epoch)
                        writer.add_scalar('Val/mAP_small', stats['coco_eval_bbox'][3], epoch)
                        writer.add_scalar('Val/mAP_medium', stats['coco_eval_bbox'][4], epoch)
                        writer.add_scalar('Val/mAP_large', stats['coco_eval_bbox'][5], epoch)
                    except Exception:
                        pass
                
                # Log per-class AP (with robust error handling)
                try:
                    if hasattr(coco_eval, 'eval') and coco_eval.eval is not None:
                        if 'precision' in coco_eval.eval:
                            precision = coco_eval.eval['precision']
                            # precision shape: [n_iou, n_points, n_cat, n_area, max_det]
                            if len(precision.shape) >= 5:
                                # Get AP for each category (mean over recall points)
                                # Use IoU 0.5:0.95 (index 0) and all areas (index 0) and max detections (index -1)
                                import numpy as np
                                per_class_ap = np.mean(precision[0, :, :, 0, -1], axis=0)  # Mean over recall thresholds
                                
                                # Get category information
                                if (hasattr(coco_eval, 'cocoGt') and coco_eval.cocoGt is not None and 
                                    hasattr(coco_eval.cocoGt, 'cats') and 
                                    hasattr(coco_eval, 'params') and coco_eval.params is not None):
                                    cats = coco_eval.cocoGt.cats
                                    cat_ids = coco_eval.params.catIds  # This is the order used in precision array
                                    
                                    # Log AP for each category (limit to avoid too many scalars)
                                    logged_count = 0
                                    max_per_class_logs = 100  # Limit to prevent TensorBoard overload
                                    for idx, cat_id in enumerate(cat_ids):
                                        if logged_count >= max_per_class_logs:
                                            break
                                        if idx < len(per_class_ap) and cat_id in cats:
                                            try:
                                                cat_info = cats[cat_id]
                                                cat_name = cat_info.get('name', f'class_{cat_id}')
                                                ap_value = float(per_class_ap[idx])
                                                # Only log if AP is valid (not NaN or Inf)
                                                if not (np.isnan(ap_value) or np.isinf(ap_value)):
                                                    writer.add_scalar(f'Val/AP_per_class/{cat_name}', ap_value, epoch)
                                                    logged_count += 1
                                            except Exception:
                                                # Skip this category if logging fails
                                                continue
                except Exception:
                    # Silently skip per-class logging if there's any error
                    pass
            except Exception:
                # Silently skip all metric logging if there's any error
                pass

    return stats, coco_evaluator
