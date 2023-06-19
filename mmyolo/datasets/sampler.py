import itertools
import random
from typing import List, Union, Optional, Iterator

import math
import torch
from mmdet.datasets import MultiSourceSampler
from mmengine.dataset import BaseDataset, DefaultSampler
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler, Dataset, DistributedSampler

from mmyolo.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class SemiSampler(MultiSourceSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_num = [len(i) for i in self.dataset.datasets]

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        while True:
            for source, num in enumerate(self.num_per_source):
                if source == 0:
                    batch_buffer_per_source = []
                    for idx in self.source2inds[source]:
                        idx += self.cumulative_sizes[source]
                        batch_buffer_per_source.append(idx)
                        if len(batch_buffer_per_source) == num:
                            batch_buffer += batch_buffer_per_source
                            break
                else:
                    batch_buffer_per_source = []
                    for k in range(num):
                        idx = random.randint(0, self.data_num[source]-1)
                        idx += self.cumulative_sizes[source]
                        batch_buffer_per_source.append(idx)
                    batch_buffer += batch_buffer_per_source

            yield from batch_buffer
            batch_buffer = []

    def __len__(self):
        return 60


@DATA_SAMPLERS.register_module()
class SemiMultiSampler(DefaultSampler):
    def __init__(self,
                 dataset,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset.datasets[0]) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset.datasets[0]) - rank) / world_size)
            self.total_size = len(self.dataset.datasets[0])

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset.datasets[0]), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset.datasets[0])).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)


@DATA_SAMPLERS.register_module()
class SemiBatchSampler(Sampler):

    def __init__(self, sampler, batch_size, sup_batch_size: int = 2, semi_batch_size: int = 1,
                 drop_last: bool = True, unsup_num=1000, start_num=1799):
        self.sampler = sampler
        self.batch_size = batch_size
        self.sup_batch_size = sup_batch_size
        self.semi_batch_size = semi_batch_size
        assert self.batch_size == self.sup_batch_size + self.semi_batch_size
        self.drop_last = drop_last
        self.unsup_num = unsup_num
        self.start_num = start_num
        assert self.drop_last is True

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.seed = sync_random_seed()
        random.seed(self.seed+self.rank)

    def __iter__(self):
        sampler_iter = iter(self.sampler)
        while True:
            try:
                batch = [
                            next(sampler_iter) for _ in range(self.sup_batch_size)
                        ] + [
                    self.start_num + random.randint(0, self.unsup_num - 1) for _ in range(self.semi_batch_size)
                ]
                # print('sample', batch)
                yield batch
            except StopIteration:
                break

    def __len__(self):
        # return 10
        return len(self.sampler) // self.sup_batch_size

