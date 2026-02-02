"""
Repeated Augmentation Sampler for PyTorch Distributed Training.

This module implements a sampler that repeats each sample multiple times within an epoch,
allowing different augmentations to be applied to the same sample. This technique can
improve training stability and generalization, especially for smaller datasets.

Reference:
    - Hoffer et al., "Augment your batch: Improving generalization through instance repetition"
    - timm library implementation
"""

import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
from typing import Optional, Iterator


class RepeatAugSampler(Sampler):
    """
    Sampler that repeats dataset samples for repeated augmentation.
    
    This sampler is designed for distributed training with data augmentation.
    Each sample in the dataset is repeated `num_repeats` times consecutively,
    allowing different random augmentations to be applied to each repetition.
    
    Args:
        dataset: Dataset to sample from.
        num_replicas: Number of processes participating in distributed training.
            If None, will be retrieved from the current distributed group.
        rank: Rank of the current process within num_replicas.
            If None, will be retrieved from the current distributed group.
        shuffle: If True, sampler will shuffle the indices.
        num_repeats: Number of times to repeat each sample. Default is 3.
        keep_original: If True, the first repetition will be marked to skip augmentation
            (requires dataset/processor support). Default is False.
        selected_round: Round the total number of samples to this value for even batching.
            Default is 256.
        seed: Random seed for shuffling. Default is 0.
    
    Example:
        >>> sampler = RepeatAugSampler(
        ...     dataset, 
        ...     num_repeats=3, 
        ...     keep_original=True
        ... )
        >>> # With num_repeats=3, each sample appears 3 times consecutively
        >>> # If keep_original=True, the first of each triplet is the original (no aug)
    """
    
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        num_repeats: int = 3,
        keep_original: bool = False,
        selected_round: int = 256,
        seed: int = 0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
                
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
                
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.keep_original = keep_original
        self.seed = seed
        self.epoch = 0
        
        # Calculate the number of samples
        self.num_samples_per_replica = int(
            math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas)
        )
        
        # Round up to selected_round for even batching
        if selected_round > 0:
            self.num_samples_per_replica = int(
                math.ceil(self.num_samples_per_replica / selected_round)
            ) * selected_round
            
        self.total_size = self.num_samples_per_replica * self.num_replicas
        
    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for one epoch.
        
        Returns indices in the format:
        - If keep_original=False: (idx, repeat_idx) pairs
        - If keep_original=True: (idx, repeat_idx, is_original) tuples
        
        However, since DataLoader expects integer indices, we return a tuple
        (sample_idx, repeat_idx) that the dataset's __getitem__ should handle.
        
        For simplicity, we return flat indices and let the dataset handle repeats
        through a different mechanism (see below).
        """
        # Deterministic shuffling based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Create repeated indices
        # Each sample is repeated num_repeats times consecutively
        repeated_indices = []
        for idx in indices:
            for repeat_idx in range(self.num_repeats):
                # Encode both the sample index and repeat index
                # We use a tuple-like encoding: (idx * num_repeats + repeat_idx)
                # This allows the dataset to decode and optionally skip augmentation
                # for the first repeat when keep_original=True
                if self.keep_original:
                    # Pass (sample_idx, repeat_idx, is_original) encoded as dict-like
                    # Since DataLoader needs int, we'll handle this differently
                    repeated_indices.append((idx, repeat_idx, repeat_idx == 0))
                else:
                    repeated_indices.append((idx, repeat_idx, False))
        
        # Pad to make it evenly divisible
        padding_size = self.total_size - len(repeated_indices)
        if padding_size > 0:
            # Pad with repeated samples from the beginning
            repeated_indices += repeated_indices[:padding_size]
        
        # Subsample for this rank
        indices_for_rank = repeated_indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices_for_rank) == self.num_samples_per_replica
        
        return iter(indices_for_rank)
    
    def __len__(self) -> int:
        return self.num_samples_per_replica
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.
        
        When `shuffle=True`, this ensures all replicas use the same permutation
        for each epoch. Otherwise, the next iteration will return the same indices.
        
        Args:
            epoch: Epoch number.
        """
        self.epoch = epoch


class RepeatAugCollator:
    """
    A wrapper collator that handles repeat augmentation indices.
    
    This collator wraps the original collate function and processes
    the (idx, repeat_idx, skip_aug) tuples from RepeatAugSampler.
    
    Args:
        base_collate_fn: The original collate function.
        dataset: The dataset being sampled.
        keep_original: Whether to skip augmentation for the first repeat.
    """
    
    def __init__(self, base_collate_fn, dataset, keep_original: bool = False):
        self.base_collate_fn = base_collate_fn
        self.dataset = dataset
        self.keep_original = keep_original
        
    def __call__(self, batch):
        """
        Process a batch of (idx, repeat_idx, skip_aug) tuples.
        
        Args:
            batch: List of tuples from RepeatAugSampler.
            
        Returns:
            Collated batch from the base collate function.
        """
        # Fetch actual samples from dataset
        samples = []
        for idx, repeat_idx, skip_aug in batch:
            # Set a flag in dataset to control augmentation
            if hasattr(self.dataset, 'set_skip_augmentation'):
                self.dataset.set_skip_augmentation(skip_aug)
            
            sample = self.dataset[idx]
            samples.append(sample)
        
        # Use original collate function
        if self.base_collate_fn is not None:
            return self.base_collate_fn(samples)
        else:
            return torch.utils.data.dataloader.default_collate(samples)


def create_repeat_aug_loader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    num_repeats: int = 3,
    keep_original: bool = False,
    collate_fn=None,
    pin_memory: bool = True,
    drop_last: bool = True,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    seed: int = 0,
):
    """
    Create a DataLoader with RepeatAugSampler.
    
    This is a convenience function to create a DataLoader with repeated
    augmentation sampling.
    
    Args:
        dataset: Dataset to load.
        batch_size: Batch size per GPU.
        num_workers: Number of data loading workers.
        num_repeats: Number of times to repeat each sample.
        keep_original: If True, first repetition skips augmentation.
        collate_fn: Custom collate function.
        pin_memory: Whether to pin memory.
        drop_last: Whether to drop the last incomplete batch.
        num_replicas: Number of distributed processes.
        rank: Rank of current process.
        seed: Random seed.
        
    Returns:
        DataLoader with RepeatAugSampler.
    """
    from torch.utils.data import DataLoader
    
    sampler = RepeatAugSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        num_repeats=num_repeats,
        keep_original=keep_original,
        seed=seed,
    )
    
    # Wrap collate function to handle repeat indices
    wrapped_collate = RepeatAugCollator(collate_fn, dataset, keep_original)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=wrapped_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    return loader, sampler
