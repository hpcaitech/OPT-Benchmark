import torch
import torch.distributed as dist

def memory_cap(size_in_GB):
    print(f"use only {size_in_GB} GB of CUDA memory")
    assert dist.is_initialized(), "memory_cap must be used after dist init"
    local_rank = dist.get_rank()
    cuda_capacity = torch.cuda.get_device_properties(local_rank).total_memory
    size_in_B = (size_in_GB * 1024**3)
    if  size_in_B > cuda_capacity:
        print(f'memory_cap is uselsess since {cuda_capacity / 1024**3} less than {size_in_GB}')
        return
    fraction = (size_in_GB * 1024**3) / cuda_capacity
    torch.cuda.set_per_process_memory_fraction(fraction, local_rank)


if __name__ == '__main__':
    memory_cap(40)