"""
Shared torch.distributed helpers for enc_pre.py / text_to_text.py /
ocr_to_text.py's full-training stages (train_encoder, run_ssl_training,
train()) -- NOT the Optuna tuning stages, which stay single-process: many
short, independent trials is a different, unrelated axis of parallelism
from data-parallel training (see each script's tune() -- untouched).

Launched via torchrun (see the .sh scripts), which sets RANK / WORLD_SIZE /
LOCAL_RANK / MASTER_ADDR / MASTER_PORT in the environment before the
script's own code runs. If none of those are set (e.g. running
`python foo.py` directly, no torchrun), init_distributed() degrades to
single-process behavior automatically -- every script here still runs
standalone exactly as before.

Why manual gradient all-reduce instead of torch.nn.parallel.
DistributedDataParallel (DDP)
------------------------------------------------------------------------
DDP's automatic gradient synchronization is implemented as hooks that fire
when the *wrapped module's* __call__ (i.e. its forward()) runs. This
project's training wrappers are never invoked that way -- they're called
through custom methods instead (TextPretrainWrapper/ImagePretrainWrapper/
OCRPretrainWrapper's .features()/.forward_pretrain(), or, in text_to_text.py
/ocr_to_text.py, several independent submodule calls: model.combine(...),
model.router(...), the text_output expert's own .forward()). Wrapping any
of these in DDP and calling them by their existing method names would
silently skip gradient sync -- training would *look* like it's running
(losses go down per-rank) while each rank actually trains on its own data
shard independently, with GPUs never averaging gradients at all. That
failure mode produces no error, just quietly-wrong (and non-reproducible)
results, which is worse than not distributing at all.

Manual all-reduce sidesteps this entirely: sync_gradients() explicitly
averages every parameter's .grad across ranks after loss.backward() and
before optimizer.step(), regardless of what method call produced that
gradient. It's less overlapped/optimized than DDP's bucketed async
communication, but for the modest world sizes this project runs at (a
handful of GPUs on one SLURM node), that overhead difference is immaterial
next to correctness.
"""
import datetime
import os

import torch
import torch.distributed as dist

# PyTorch's default process-group timeout is 30 minutes -- it governs both
# collective ops (all_reduce/barrier) *and* the TCPStore rendezvous every
# collective's setup goes through. That's far too short for this project's
# "tune" stages: they run single-process on rank 0 only (see this module's
# docstring), with every *other* rank sitting idle at a barrier waiting for
# rank 0 to finish -- and a real Optuna search across several trials,
# each doing real training + validation, routinely takes well over 30
# minutes. Rank 0 being legitimately busy (not stuck) still trips the
# other ranks' default timeout, killing the whole job with a "Socket
# Timeout" that looks like a hang but isn't one. A generous, explicit
# timeout (comfortably inside the 48-hour SLURM allocation these jobs run
# under) fixes this without needing to redesign how ranks stay in sync.
_PROCESS_GROUP_TIMEOUT = datetime.timedelta(hours=24)


def init_distributed():
    """Call once, near the top of main(), before any CUDA context is
    created. Returns (rank, world_size, local_rank, is_distributed,
    device)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return 0, 1, 0, False, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend, device = "nccl", f"cuda:{local_rank}"
    else:
        backend, device = "gloo", "cpu"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                            timeout=_PROCESS_GROUP_TIMEOUT)
    return rank, world_size, local_rank, True, device


def cleanup_distributed(is_distributed):
    if is_distributed:
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def barrier(is_distributed):
    if is_distributed:
        dist.barrier()


def rank0_first(rank, world_size, fn, *args, **kwargs):
    """Call fn(*args, **kwargs) on rank 0 first, then have every other rank
    wait at a barrier before calling it too. Use this for anything that
    downloads/caches to a shared on-disk path (dataset preparation, in
    particular) -- letting every rank race to populate the same cache
    directory concurrently is exactly the kind of corruption this project's
    checkpoint code was hardened against (see enc_pre.py's
    _atomic_torch_save); dataset caching has the same failure mode.
    Every rank still calls fn and uses its own return value (rank 0's
    result isn't broadcast) -- by the time non-zero ranks call it, the
    cache is already populated, so their call is a cheap local read."""
    if world_size <= 1:
        return fn(*args, **kwargs)
    if is_main_process(rank):
        result = fn(*args, **kwargs)
    barrier(True)
    if not is_main_process(rank):
        result = fn(*args, **kwargs)
    barrier(True)
    return result


def shard_indices(order, rank, world_size):
    """Stride-slice a (already globally shuffled, identical-on-every-rank)
    index list across ranks: rank r gets order[r::world_size]. Truncates to
    a multiple of world_size first so every rank's shard is exactly the
    same length -- required so every rank calls sync_gradients() /
    optimizer.step() the same number of times per epoch; otherwise a rank
    with one extra batch would call a collective (all_reduce) that the
    other ranks never issue, and the whole job would hang waiting on it."""
    usable = len(order) - (len(order) % world_size)
    return order[:usable][rank::world_size]


def sync_gradients(parameters, world_size):
    """Manual data-parallel gradient averaging (see module docstring for
    why this is used instead of DDP): all-reduce (sum) every parameter's
    gradient across ranks, then divide by world_size. Call after
    loss.backward(), before optimizer.step().

    Every rank must issue the exact same sequence of all_reduce calls --
    the same hazard shard_indices' docstring already documents for batch
    counts applies per-parameter here too, and this used to skip the
    all_reduce entirely for any parameter whose .grad was None on a given
    rank. That's correct for a param frozen everywhere (requires_grad=
    False; every rank agrees and skips it, no mismatch) but silently wrong
    for a trainable one some rank's particular slice of this step's data
    happened not to touch -- .grad is legitimately None there (nothing
    backpropagated to it) while another rank's different data *did* touch
    it, so that rank calls all_reduce and this one doesn't: a real
    ChildFailedError/NCCL ALLREDUCE deadlock, observed in production
    (slurm-4733884 -- every rank piled up on the same collective for the
    full 24h process-group timeout before aborting). A zero tensor stands
    in for a trainable param's missing grad so every rank always issues
    the same call for it -- functionally a no-op contribution to the sum,
    but keeps every rank's collective sequence identical regardless of
    which rows happened to touch what."""
    if world_size <= 1:
        return
    for p in parameters:
        if p.grad is None:
            if not p.requires_grad:
                continue
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size


def broadcast_parameters(parameters, src=0):
    """Broadcast every parameter's data from rank ``src`` to all other
    ranks -- call once after model construction (and again after loading a
    resume checkpoint) so every rank starts an epoch from bit-identical
    weights. Without this, ranks that didn't load the checkpoint (or that
    initialised with a different random seed) would diverge from the start,
    and sync_gradients() alone does not fix already-divergent parameters --
    it only keeps identically-initialised parameters in sync going forward."""
    for p in parameters:
        dist.broadcast(p.data, src=src)
