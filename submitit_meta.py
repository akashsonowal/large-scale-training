## SLURM using submitit
import submitit

def add(a, b):
    return a + b

# ask for resources
executor = submitit.AutoExecutor(folder="my_shared_folder")
executor.update_parameters(gpus_per_node=2)

# submit to the cluster
job = executor.submit(add, 5, 7)  # will compute add(5, 7)

# waits for completion and returns output
output = job.result()
# 5 + 7 = 12...  your addition was computed in the cluster
assert output == 12  

## This uses MPI for communication