# large-scale-training

Job scheduling for clusters using SLURM

Multinode training involves deploying a training job across several machines. There are two ways to do this:

running a torchrun command on each machine with identical rendezvous arguments, or

deploying it on a compute cluster using a workload manager (like SLURM)
