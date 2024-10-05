# large-scale-training

Job scheduling for clusters using SLURM

Multinode training involves deploying a training job across several machines. There are two ways to do this:

running a torchrun command on each machine with identical rendezvous arguments, or

deploying it on a compute cluster using a workload manager (like SLURM)

## Single Node multigpu
```
torchrun --standalone --nproc_per_node=gpu multigpu_torchrun.py 50 10
```

## Multinode
ifconfig to see all network interface
```
export NCCL_SOCKET_IFNAME=ens5
cd distributed/
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --rdzv_id=576 --rdzv_backend=c10d --rdzv_endpoint=172.31.2.176:29603 multinode_torchrun.py 50 10
```