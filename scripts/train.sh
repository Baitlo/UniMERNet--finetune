#!/bin/bash

# train unimernet_base
HF_ENDPOINT=https://hf-mirror.com
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cfg-path /home/ubuntu/baiweikang/Projects/Taichi/UniMERNet-main/configs/train/unimernet_base_with_aug_repeats.yaml

# train unimernet_small
# python -m torch.distributed.launch --nproc_per_node=8 train.py --cfg-path configs/train/unimernet_small_encoder6666_decoder8_dim768.yaml

# train unimernet_tiny
# python -m torch.distributed.launch --nproc_per_node=8 train.py --cfg-path configs/train/unimernet_tiny_encoder6666_decoder8_dim768.yaml