def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All CUDA operations are completed (synchronize)
    """
    import torch

    torch.distributed.barrier()
    torch.cuda.synchronize()


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.utils import StatelessProcessGroup
    import os
    
    # Check if we should use Gloo instead of NCCL
    use_gloo = os.environ.get('USE_GLOO_FOR_VLLM', 'false').lower() == 'true'
    
    if use_gloo:
        # Use Gloo backend for communication
        import torch.distributed as dist
        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://{master_address}:{master_port}',
            rank=rank,
            world_size=world_size
        )
        return dist.group.WORLD
    else:
        # Use NCCL (original behavior)
        try:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
            pynccl = PyNcclCommunicator(pg, device=device)
            return pynccl
        except Exception as e:
            print(f"NCCL initialization failed: {e}")
            print("Falling back to Gloo backend...")
            # Fallback to Gloo
            import torch.distributed as dist
            dist.init_process_group(
                backend='gloo',
                init_method=f'tcp://{master_address}:{master_port}',
                rank=rank,
                world_size=world_size
            )
            return dist.group.WORLD
