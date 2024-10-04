import os 
from typing import Callable 

import torch 
import torch.distributed as dist 

import multiprocessing as mp # for spawning processes in a CPU

def init_process(rank: int, size: int, fn: Callable[[int, int], None], backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def func(rank: int, size: int):
    print(f"Say Hi! {rank}")

def do_reduce(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
    print(f"[{rank}] data = {tensor[0]}")

def all_reduce(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"{rank} data = {tensor[0]}")

def scatter(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.empty(1)

    if rank == 0:
        tensor_list = [torch.ones([i + 1], dtype=torch.float32) for i in range(size)]
        dist.scatter(tensor, scatter_list=tensor_list, src=0, group=group)
    else:
        dist.scatter(tensor, scatter_list=[], src=0, group=group)
    
    print(f"[{rank}] data = {tensor[0]}")

def do_gather(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.tensor([rank], dtype=torch.float32)

    if rank == 0:
        tensor_list = [torch.empty(1) for i in range(size)]
        dist.gather(tensor, gather_list=tensor_list, dst=0, group=group)
    else:
        dist.gather(tensor, gather_list=[], dst=0, group=group)
    
    if rank == 0:
        print(f"[{rank}] data = {tensor_list}")

def do_all_gather(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.tensor([rank], dtype=torch.float32)
    tensor_list = [torch.empty(1) for i in range(size)]
    dist.all_gather(tensor_list, tensor, group=group)
    print(f"[{rank}] data = {tensor_list}")

def do_broadcast(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    if rank == 0:
        tensor = torch.tensor([rank], dtype=torch.float32)
    else:
        tensor = torch.empty(1)
    
    dist.broadcast(tensor, src=0, group=group)
    print(f"{[rank]} data = {tensor}")

if __name__ =="__main__":
    size = 4
    processes = []

    mp.set_start_method("spawn")

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, do_broadcast))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()