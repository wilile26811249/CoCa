import json
import logging
import math
import os
import time
from contextlib import suppress

import torch

from coca_cfg import CLIPTextCfg, CLIPVisionCfg, MultimodalCfg
from coca_model import _build_vision_tower, _build_text_tower, _build_text_decoder_tower
from coca_model import CoCa


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, dataloader, loss, epoch, optimizer, scaler):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    autocast = get_autocast('')
    input_dtype = get_input_dtype('')


    model.train()

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        images, texts = batch
        images = images.to(device = device, dtype = input_dtype, non_blocking = True)
        texts = texts.to(device = device, non_blocking = True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            model_out = model(images, texts)
            logit_scale = model_out["logit_scale"]
            losses = loss(**model_out, output_dict = True)

            total_loss = sum(losses.values())
            losses["loss"] = total_loss

        backward(total_loss, scaler)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        batch_size = len(images)
        num_samples = batch_count * batch_size
        percent_complete = 100.0 * batch_count / len(dataloader)

        # NOTE loss is coarsely sampled, just master node and per log update
        for key, val in losses.items():
            if key not in losses_m:
                losses_m[key] = AverageMeter()
            losses_m[key].update(val.item(), batch_size)

        logit_scale_scalar = logit_scale.item()
        loss_log = " ".join(
            [
                f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                for loss_name, loss_m in losses_m.items()
            ]
        )
        samples_per_second = batch_size / batch_time_m.val
        samples_per_second_per_gpu = batch_size / batch_time_m.val

        logging.info(
            f"Train Epoch: {epoch}"
            f"Data (t): {data_time_m.avg:.3f} "
            f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
            f"LR: {optimizer.param_groups[0]['lr']:5f} "
            f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
        )

        print(f"Train Epoch: {epoch}, Data (t): {data_time_m.avg:.3f}")
        for k, v in losses_m.items():
            print(f"{k}: {v.avg:.3f},",  end = '')
        print()