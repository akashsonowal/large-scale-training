import os 
from typing import Callable

import torch
import torch.distributed as dist 

import multiprocessing as mp # for spawning the process

def init_process(rank: int, size: int, fn: Callable[[int, int], None], backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def func(rank: int, size: int):
    continue

if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, func))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()