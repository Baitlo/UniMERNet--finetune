#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UniMERNet DDP Inference Script for CMER_BENCH

This script performs distributed inference on the CMER_BENCH dataset
with three difficulty levels: EASY, MODERATE, COMPLEX.

Usage:
    # Single GPU
    python infer_ddp.py --cfg-path configs/infer_ddp.yaml --output infer_results.jsonl

    # Multi-GPU DDP
    torchrun --nproc_per_node=8 infer_ddp.py --cfg-path configs/infer_ddp.yaml --output infer_results.jsonl
"""

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

import unimernet.tasks as tasks
from unimernet.common.config import Config
from unimernet.common.dist_utils import init_distributed_mode, is_main_process, get_rank, get_world_size
from unimernet.datasets.builders import *
from unimernet.models import *
from unimernet.processors import *
from unimernet.tasks import *
from unimernet.processors import load_processor


class CMERBenchDataset(Dataset):
    """Dataset for CMER_BENCH evaluation.
    
    Args:
        data_root: Root directory containing the three difficulty level folders
        categories: List of categories to include, e.g., ['easy', 'moderate', 'complex']
        transform: Optional transform to apply to images
    """
    
    CATEGORY_FOLDERS = {
        'easy': 'CMER_BENCH_1_0__EASY_for_unimer',
        'moderate': 'CMER_BENCH_1_0__MODERATE_for_unimer',
        'complex': 'CMER_BENCH_1_0__COMPLEX_for_unimer',
    }
    
    def __init__(self, data_root, categories=None, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        
        if categories is None:
            categories = ['easy', 'moderate', 'complex']
        
        self.samples = []  # List of (image_path, original_id, category)
        
        for category in categories:
            folder_name = self.CATEGORY_FOLDERS.get(category.lower())
            if folder_name is None:
                print(f"Warning: Unknown category '{category}', skipping...")
                continue
            
            category_path = self.data_root / folder_name
            if not category_path.exists():
                print(f"Warning: Category path does not exist: {category_path}, skipping...")
                continue
            
            # Load mapping.json
            mapping_file = category_path / 'mapping.json'
            if not mapping_file.exists():
                print(f"Warning: mapping.json not found in {category_path}, skipping...")
                continue
            
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_list = json.load(f)
            
            # Create mapping: new_name -> old_id
            name_to_id = {item['new_name']: item['old_id'] for item in mapping_list}
            
            # Load all images in the folder
            images_dir = category_path / 'images'
            if not images_dir.exists():
                print(f"Warning: images folder not found in {category_path}, skipping...")
                continue
            
            for img_file in sorted(images_dir.iterdir()):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    original_id = name_to_id.get(img_file.name)
                    if original_id is None:
                        print(f"Warning: No mapping found for {img_file.name}, using filename as id")
                        original_id = img_file.stem
                    
                    self.samples.append((
                        str(img_file),
                        original_id,
                        category.lower()
                    ))
        
        print(f"Loaded {len(self.samples)} samples from {len(categories)} categories")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, original_id, category = self.samples[idx]
        
        # Load and transform image
        raw_image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(raw_image)
        else:
            image = raw_image
        
        return image, original_id, category, idx


def collate_fn(batch):
    """Custom collate function to handle mixed data types."""
    images = torch.stack([item[0] for item in batch])
    original_ids = [item[1] for item in batch]
    categories = [item[2] for item in batch]
    indices = [item[3] for item in batch]
    return images, original_ids, categories, indices


def setup_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="UniMERNet DDP Inference")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--data-root", type=str, 
                        default="/home/ubuntu/bigdiskdata/baiweikang/CMER_BENCH_1_0_for_unimer",
                        help="Root directory of CMER_BENCH dataset.")
    parser.add_argument("--output", type=str, default="infer_results.jsonl",
                        help="Output JSONL file path.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers.")
    parser.add_argument("--categories", type=str, nargs='+', 
                        default=['easy', 'moderate', 'complex'],
                        help="Categories to evaluate (easy, moderate, complex).")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override config settings in xxx=yyy format.",
    )
    # Distributed training arguments (required by init_distributed_mode)
    parser.add_argument("--dist-url", type=str, default="env://",
                        help="URL for distributed training.")
    parser.add_argument("--dist-backend", type=str, default="nccl",
                        help="Distributed backend.")
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize distributed mode
    init_distributed_mode(args)
    
    setup_seeds(42 + get_rank())
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = get_rank()
    world_size = get_world_size()
    
    if is_main_process():
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Data root: {args.data_root}")
        print(f"Output file: {args.output}")
        print(f"Categories: {args.categories}")
    
    # Load model configuration
    cfg = Config(args)
    
    if is_main_process():
        print(f"arch_name: {cfg.config.model.arch}")
        print(f"model_type: {cfg.config.model.model_type}")
        if hasattr(cfg.config.model, 'finetuned'):
            print(f"checkpoint: {cfg.config.model.finetuned}")
        print("=" * 80)
    
    # Build model
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model.to(device)
    model.eval()
    
    # Load visual processor
    vis_processor = load_processor(
        'formula_image_eval', 
        cfg.config.datasets.formula_rec_eval.vis_processor.eval
    )
    
    # Create dataset
    transform = transforms.Compose([vis_processor])
    dataset = CMERBenchDataset(
        data_root=args.data_root,
        categories=args.categories,
        transform=transform
    )
    
    # Create distributed sampler and dataloader
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    if is_main_process():
        print(f"Total samples: {len(dataset)}")
        print(f"Batches per GPU: {len(dataloader)}")
        print("Starting inference...")
    
    # Inference
    local_results = []  # List of (idx, id, tex, category)
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc=f"GPU {rank}") if is_main_process() else dataloader
        for images, original_ids, categories, indices in iterator:
            images = images.to(device)
            
            # Generate predictions
            output = model.generate({"image": images})
            predictions = output["pred_str"]
            
            # Store results with indices
            for idx, orig_id, tex, cat in zip(indices, original_ids, predictions, categories):
                local_results.append({
                    'idx': idx,
                    'id': orig_id,
                    'tex': tex,
                    'category': cat
                })
    
    # Synchronize all processes
    if world_size > 1:
        dist.barrier()
    
    # Gather results from all processes
    if world_size > 1:
        # Save local results to temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"infer_results_rank{rank}.json")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(local_results, f)
        
        dist.barrier()
        
        # Main process collects all results
        if is_main_process():
            all_results = []
            for r in range(world_size):
                temp_file_r = os.path.join(temp_dir, f"infer_results_rank{r}.json")
                with open(temp_file_r, 'r', encoding='utf-8') as f:
                    results_r = json.load(f)
                all_results.extend(results_r)
                # Clean up temp file
                os.remove(temp_file_r)
            
            # Sort by original index to maintain order
            all_results.sort(key=lambda x: x['idx'])
            
            # Remove duplicate indices (in case of padding in distributed sampling)
            seen_indices = set()
            unique_results = []
            for item in all_results:
                if item['idx'] not in seen_indices:
                    seen_indices.add(item['idx'])
                    unique_results.append(item)
            
            # Verify count
            if len(unique_results) != len(dataset):
                print(f"Warning: Result count ({len(unique_results)}) != dataset size ({len(dataset)})")
            
            # Save to JSONL
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in unique_results:
                    # Remove idx from output
                    output_item = {
                        'id': item['id'],
                        'tex': item['tex'],
                        'category': item['category']
                    }
                    f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
            
            print(f"\nResults saved to: {output_path}")
            print(f"Total samples: {len(unique_results)}")
            
            # Print category statistics
            category_counts = {}
            for item in unique_results:
                cat = item['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            print("Category statistics:")
            for cat, count in sorted(category_counts.items()):
                print(f"  {cat}: {count}")
        
        dist.barrier()
    else:
        # Single GPU - just save directly
        all_results = sorted(local_results, key=lambda x: x['idx'])
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in all_results:
                output_item = {
                    'id': item['id'],
                    'tex': item['tex'],
                    'category': item['category']
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        
        print(f"\nResults saved to: {output_path}")
        print(f"Total samples: {len(all_results)}")
        
        # Print category statistics
        category_counts = {}
        for item in all_results:
            cat = item['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print("Category statistics:")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
