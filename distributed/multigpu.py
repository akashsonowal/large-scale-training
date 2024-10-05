import os
import torch
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from data import MyTrainDataset

import multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def setup_ddp(rank, world_size):
    pass 

if __name__ == "__main__":
    import argparse
    world_size = torch.cuda.device_count()
    print(world_size)