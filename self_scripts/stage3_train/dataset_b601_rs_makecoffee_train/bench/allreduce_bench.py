"""直接实测：本模型梯度(206MB)在本机拓扑上的 allreduce 耗时"""
import os, time, torch, torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank(); world = dist.get_world_size()
torch.cuda.set_device(rank)

N_ELEM = 51_599_239                       # 本模型可训参数量
buf = torch.ones(N_ELEM, dtype=torch.float32, device=f"cuda:{rank}")
MB = N_ELEM * 4 / 1024**2

# warmup
for _ in range(5): dist.all_reduce(buf)
torch.cuda.synchronize()

ITERS = 30
dist.barrier(); t0 = time.perf_counter()
for _ in range(ITERS):
    dist.all_reduce(buf)
torch.cuda.synchronize()
t1 = time.perf_counter()

per = (t1 - t0) / ITERS
# ring allreduce 每 rank 实际搬运量 = 2*(N-1)/N * size
vol = 2 * (world - 1) / world * MB
if rank == 0:
    print(f"RESULT world={world} size={MB:.1f}MB  allreduce={per*1000:.2f}ms  "
          f"每rank搬运={vol:.1f}MB  有效带宽={vol/1024/(per):.2f}GB/s")
dist.destroy_process_group()
