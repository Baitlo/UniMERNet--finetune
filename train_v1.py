import argparse
import logging
import os
import random
from math import ceil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from unimernet.common.config import Config
from unimernet.common.dist_utils import get_rank, get_world_size, init_distributed_mode
from unimernet.common.logger import setup_logger
from unimernet.common.registry import registry
from unimernet.common.utils import now
from unimernet.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from unimernet.datasets.data_utils import concat_datasets, prepare_sample, reorg_datasets_by_split

# imports modules for registration
from unimernet.datasets.builders import *  # noqa: F403
from unimernet.models import *  # noqa: F403
from unimernet.processors import *  # noqa: F403

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDP training entrypoint for UnimerNet (from scratch)"
    )
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--dist-url",
        default=None,
        help="override distributed init url (default from config).",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help=(
            "override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead."
        ),
    )

    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def build_datasets(cfg):
    datasets = {}
    for dataset_name in cfg.datasets_cfg:
        builder_cls = registry.get_builder_class(dataset_name)
        builder = builder_cls(cfg.datasets_cfg[dataset_name])
        datasets[dataset_name] = builder.build_datasets()

    datasets_by_split = reorg_datasets_by_split(datasets)
    return concat_datasets(datasets_by_split)


def build_model(cfg):
    model_cls = registry.get_model_class(cfg.model_cfg.arch)
    if model_cls is None:
        raise ValueError(f"Unknown model architecture: {cfg.model_cfg.arch}")

    cfg.model_cfg.load_pretrained = False
    cfg.model_cfg.load_finetuned = False

    return model_cls.from_config(cfg.model_cfg)


def build_optimizer(cfg, model):
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "ln" in name or "bn" in name:
            p_non_wd.append(param)
        else:
            p_wd.append(param)
        num_parameters += param.data.nelement()

    logging.info("number of trainable parameters: %d", num_parameters)
    optim_params = [
        {"params": p_wd, "weight_decay": float(cfg.run_cfg.weight_decay)},
        {"params": p_non_wd, "weight_decay": 0},
    ]
    beta2 = cfg.run_cfg.get("beta2", 0.999)
    return torch.optim.AdamW(
        optim_params,
        lr=float(cfg.run_cfg.init_lr),
        weight_decay=float(cfg.run_cfg.weight_decay),
        betas=(0.9, beta2),
    )


def build_lr_scheduler(cfg, optimizer, iters_per_epoch):
    lr_sched = cfg.run_cfg.lr_sched
    max_epoch = cfg.run_cfg.get("max_epoch", 1)
    min_lr = cfg.run_cfg.min_lr
    init_lr = cfg.run_cfg.init_lr
    warmup_lr = cfg.run_cfg.get("warmup_lr", -1)
    warmup_steps = cfg.run_cfg.get("warmup_steps", 0)
    decay_rate = cfg.run_cfg.get("lr_decay_rate", None)

    if lr_sched == "linear_warmup_step_lr":
        return LinearWarmupStepLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=decay_rate,
            warmup_start_lr=warmup_lr,
            warmup_steps=warmup_steps,
        )

    if lr_sched == "linear_warmup_cosine_lr":
        return LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            warmup_start_lr=warmup_lr,
            warmup_steps=warmup_steps,
            iters_per_epoch=iters_per_epoch,
        )

    raise ValueError(f"Unknown lr scheduler: {lr_sched}")


def create_train_loader(cfg, train_dataset):
    sampler = None
    if cfg.run_cfg.distributed:
        sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )

    return DataLoader(
        train_dataset,
        batch_size=cfg.run_cfg.batch_size_train,
        num_workers=cfg.run_cfg.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=getattr(train_dataset, "collater", None),
        drop_last=True,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    lr_scheduler,
    scaler,
    device,
    epoch,
    accum_grad_iters,
    log_freq,
):
    model.train()
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)

    total_loss = 0.0
    for step, samples in enumerate(loader):
        samples = prepare_sample(samples, cuda_enabled=device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            loss = outputs["loss"] / accum_grad_iters

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_grad_iters == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        lr_scheduler.step(cur_epoch=epoch, cur_step=step)

        total_loss += loss.item() * accum_grad_iters

        if (step + 1) % log_freq == 0 and get_rank() == 0:
            logging.info(
                "Epoch %d step %d/%d - loss: %.6f",
                epoch,
                step + 1,
                len(loader),
                total_loss / (step + 1),
            )

    return total_loss / max(len(loader), 1)


def main():
    job_id = now()

    args = parse_args()
    cfg = Config(args)

    if args.dist_url is not None:
        cfg.run_cfg.dist_url = args.dist_url

    if "resume_ckpt_path" in cfg.run_cfg:
        cfg.run_cfg.resume_ckpt_path = None

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()

    if get_rank() == 0:
        logging.info("Job ID: %s", job_id)
    cfg.pretty_print()

    datasets = build_datasets(cfg)
    train_splits = cfg.run_cfg.get("train_splits", ["train"])
    train_split = train_splits[0] if len(train_splits) > 0 else "train"
    train_dataset = datasets.get(train_split)
    if train_dataset is None:
        raise ValueError(
            f"No training dataset found for split '{train_split}'. Check train_splits and dataset configs."
        )

    train_loader = create_train_loader(cfg, train_dataset)
    iters_per_epoch = len(train_loader)

    model = build_model(cfg)
    device = torch.device(cfg.run_cfg.device)
    model = model.to(device)
    if cfg.run_cfg.distributed:
        model = DDP(model, device_ids=[cfg.run_cfg.gpu], find_unused_parameters=False)

    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer, iters_per_epoch)
    scaler = torch.cuda.amp.GradScaler() if cfg.run_cfg.get("amp", False) else None

    accum_grad_iters = int(cfg.run_cfg.get("accum_grad_iters", 1))
    log_freq = int(cfg.run_cfg.get("log_freq", 50))

    max_iters = cfg.run_cfg.get("max_iters", None)
    if max_iters is not None:
        max_iters = int(max_iters)
        max_epoch = max(1, ceil(max_iters / max(iters_per_epoch, 1)))
    else:
        max_epoch = int(cfg.run_cfg.get("max_epoch", 1))

    total_iters = 0
    for epoch in range(max_epoch):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            scaler,
            device,
            epoch,
            accum_grad_iters,
            log_freq,
        )
        total_iters += iters_per_epoch
        if get_rank() == 0:
            logging.info("Epoch %d finished. Avg loss: %.6f", epoch, avg_loss)

        if max_iters is not None and total_iters >= max_iters:
            break

    if cfg.run_cfg.distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
