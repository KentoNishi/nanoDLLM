import os
import sys
import uuid
import time
import glob
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from muon import Muon
from model import BlockGPT, BlockGPTConfig


code = "\n".join([
    open(__file__).read(),
    open("model.py").read()
])

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# -----------------------------------------------------------------------------
# Parameters


@dataclass
class Hyperparameters:
    train_files = "data/finewebedu10B/finewebedu_train_*.bin"
    val_files = "data/finewebedu10B/finewebedu_val_*.bin"
    val_tokens = 10_485_760
    train_seq_len = 4 * 1024
    val_seq_len = 8 * 1024
    grad_accum_steps_per_device = 1
    num_iterations = 10_000
    cooldown_frac = 0.8
    vocab_size = 50_257
    val_loss_every = 10
    save_checkpoint = True
    dataset_mode = "fineweb"  # options: fineweb, combined, combined_guidance, cbt, easymath
    run_id: str | None = None
    resume_from: str | None = None
    pretrained_checkpoint: str | None = None


def parse_cli_overrides():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset_mode', choices=[
        'fineweb', 'combined', 'combined_guidance', 'cbt', 'easymath'
    ], default=None)
    parser.add_argument('--run_id', default=None)
    parser.add_argument('--resume_from', default=None)
    parser.add_argument('--pretrained_checkpoint', default=None)
    parser.add_argument('--train_seq_len', type=int, default=None)
    parser.add_argument('--val_seq_len', type=int, default=None)
    parser.add_argument('--grad_accum_steps_per_device', type=int, default=None)
    parser.add_argument('--num_iterations', type=int, default=None)
    parser.add_argument('--val_loss_every', type=int, default=None)
    parser.add_argument('--val_tokens', type=int, default=None)
    parser.add_argument('--cooldown_frac', type=float, default=None)
    parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true')
    parser.add_argument('--no-save_checkpoint', dest='save_checkpoint', action='store_false')
    parser.set_defaults(save_checkpoint=None)
    parser.add_argument('--local_rank', type=int, default=None)
    cli_args, _ = parser.parse_known_args()
    return cli_args


# -----------------------------------------------------------------------------
# Simple Distributed Data Loader


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=True
        )
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens


def distributed_data_generator(
    filename_pattern: str,
    batch_size: int,
    rank: int,
    world_size: int,
):
    files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_bs = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0

    while True:
        if pos + batch_size >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_bs:][:local_bs]
        inputs = buf.to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs


def unpack_sample(sample):
    if isinstance(sample, tuple) and len(sample) == 2:
        return sample
    return sample, None


def course_data_generator(loader, device):
    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        seq = batch["input_ids"].squeeze(0).to(device=device, dtype=torch.int64, non_blocking=True)
        guidance = batch.get("guidance_id")
        if guidance is not None:
            guidance = guidance.squeeze(0).to(device=device, dtype=torch.long, non_blocking=True)
        yield seq, guidance


# -----------------------------------------------------------------------------
# Main


def evaluate(model, loader, steps):
    """Run model in eval mode over `steps` batches from `loader`."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(steps):
            x, guidance = unpack_sample(next(loader))
            total += model(x, guidance)
    return total / steps


def train_step(model, loader, step, optimizers, optimizer2, accum_steps):
    # forward/backward accumulation
    for _ in range(accum_steps):
        x, guidance = unpack_sample(next(loader))
        loss = model(x, guidance)
        loss.backward()

    # gradient all‐reduce across ranks
    for _, p in model.named_parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    # adjust learning rates
    lr = get_lr(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lr

    # Muon momentum warmup
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1.0)
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

    # step and clear
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)


cli_overrides = parse_cli_overrides()
args = Hyperparameters()

def _override(field):
    value = getattr(cli_overrides, field, None)
    if value is not None:
        setattr(args, field, value)

for name in [
    'dataset_mode', 'run_id', 'resume_from', 'pretrained_checkpoint',
    'train_seq_len', 'val_seq_len', 'grad_accum_steps_per_device',
    'num_iterations', 'val_loss_every', 'val_tokens', 'cooldown_frac'
]:
    _override(name)

if getattr(cli_overrides, 'save_checkpoint', None) is not None:
    args.save_checkpoint = cli_overrides.save_checkpoint

guidance_enabled = args.dataset_mode == 'combined_guidance'
is_finetune = args.dataset_mode != 'fineweb'

if is_finetune:
    if args.num_iterations > 250:
        args.num_iterations = 250
    if args.val_loss_every > 5:
        args.val_loss_every = 5

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = rank == 0

default_run_id = args.run_id or os.environ.get("RUN_ID")
if default_run_id is None:
    default_run_id = "fineweb-base" if not is_finetune else f"{args.dataset_mode}-finetune"
run_id = default_run_id or str(uuid.uuid4())
logs_root = Path("logs")
run_dir = logs_root / run_id
logfile = str(run_dir / "events.txt")
if master_process:
    run_dir.mkdir(parents=True, exist_ok=True)
    print(logfile)
resume_path = args.resume_from
if resume_path is None:
    candidate = run_dir / "state_latest.pt"
    if candidate.exists():
        resume_path = str(candidate)
pretrained_path = args.pretrained_checkpoint or os.environ.get("DLLM_PRETRAINED_CHECKPOINT")
if pretrained_path is None:
    default_base = logs_root / "fineweb-base" / "state_latest.pt"
    if default_base.exists():
        pretrained_path = str(default_base)


def print0(s: str, console: bool = False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)


# dump the training script
print0(code)
print0("=" * 100)
print0(f"Python {sys.version}")
print0(f"PyTorch {torch.version.__version__}")
print0(os.popen("nvidia-smi").read())
print0("=" * 100)


model_config = BlockGPTConfig(
    num_guidance_tokens=2 if guidance_enabled else 0,
    dropout=0.3 if is_finetune else 0.0,
)
model = BlockGPT(model_config).cuda()

for m in model.modules():
    if isinstance(m, torch.nn.Embedding):
        m.bfloat16()

for p in model.parameters():
    dist.broadcast(p.detach(), 0)


hidden_matrix_params = [
    p
    for n, p in model.blocks.named_parameters()
    if p.ndim >= 2 and "embed" not in n
]
guidance_params = []
embed_params = []
for name, param in model.named_parameters():
    if "guidance_embed" in name:
        guidance_params.append(param)
    elif "embed" in name:
        embed_params.append(param)
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

if is_finetune:
    head_lr = embed_lr = scalar_lr = 6e-5
    guidance_lr = 6e-4
    muon_lr = 6e-5
else:
    head_lr = 0.0011
    embed_lr = 0.06
    scalar_lr = 0.04
    guidance_lr = 0.06
    muon_lr = 0.025

adam_params = [
    dict(params=head_params, lr=head_lr),
    dict(params=embed_params, lr=embed_lr),
    dict(params=scalar_params, lr=scalar_lr),
]
if guidance_params:
    adam_params.append(dict(params=guidance_params, lr=guidance_lr))
optimizer1 = torch.optim.Adam(
    adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True
)
optimizer2 = Muon(
    hidden_matrix_params,
    lr=muon_lr,
    momentum=0.95,
)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

start_step = 0
best_val_loss = float("inf")

def _load_state_dict(path: str, load_optim: bool):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if master_process:
        if missing:
            print0(f"Missing keys when loading {path}: {missing}")
        if unexpected:
            print0(f"Unexpected keys when loading {path}: {unexpected}")
    if load_optim and isinstance(checkpoint, dict) and 'optimizers' in checkpoint:
        for opt, state in zip(optimizers, checkpoint['optimizers']):
            opt.load_state_dict(state)
    return checkpoint

if resume_path:
    ckpt = _load_state_dict(resume_path, load_optim=True)
    start_step = ckpt.get('step', 0) + 1
    best_val_loss = ckpt.get('best_val_loss', best_val_loss)
    if master_process:
        print0(f"Resumed training from {resume_path} at step {start_step}")
elif is_finetune:
    if not pretrained_path or not Path(pretrained_path).exists():
        raise FileNotFoundError(
            "A pretrained checkpoint is required for finetuning. "
            "Set `pretrained_checkpoint` or environment variable DLLM_PRETRAINED_CHECKPOINT."
        )
    _load_state_dict(pretrained_path, load_optim=False)
    if master_process:
        print0(f"Initialized weights from pretrained checkpoint {pretrained_path}")


def persist_checkpoint(step: int, final: bool = False):
    if not master_process or not args.save_checkpoint:
        return
    ckpt = dict(
        step=step,
        code=code,
        model=model.state_dict(),
        optimizers=[opt.state_dict() for opt in optimizers],
        best_val_loss=best_val_loss,
        dataset_mode=args.dataset_mode,
        run_id=run_id,
    )
    latest_path = run_dir / "state_latest.pt"
    torch.save(ckpt, latest_path)
    if final:
        final_path = run_dir / f"state_step{step:06d}.pt"
        torch.save(ckpt, final_path)


def get_lr(step: int) -> float:
    x = step / args.num_iterations
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    w = (1 - x) / args.cooldown_frac
    return w * 1.0 + (1 - w) * 0.1


model = torch.compile(model, dynamic=False)

# Data selection: fineweb (default) or course datasets
use_fineweb = args.dataset_mode == "fineweb"
course_val_steps = None
course_seq_len = None

if use_fineweb:
    train_loader = distributed_data_generator(
        args.train_files,
        world_size * args.train_seq_len,
        rank,
        world_size,
    )
else:
    try:
        import cs2420_cs2823r_final_project as cs_project  # type: ignore
        from cs2420_cs2823r_final_project.data.combined_dataset import CombinedDataset, CombinedDatasetForGuidance  # type: ignore
        from cs2420_cs2823r_final_project.data.cbt_dataset import CBTDataset  # type: ignore
        from cs2420_cs2823r_final_project.data.easymath_dataset import EasyMathDataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Install the `conditional_dllm_class_project` package (from cs2420_cs2823r_final_project/) "
            f"before using dataset_mode '{args.dataset_mode}'."
        ) from exc

    from datasets import load_from_disk  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    from torch.utils.data import DataLoader, random_split

    data_root = Path(cs_project.__file__).resolve().parent / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"Could not locate course data directory under {data_root}")

    cbt_data = load_from_disk(str(data_root / "cbt"))
    easymath_data = load_from_disk(str(data_root / "easymath"))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    block_size = 64
    if args.dataset_mode == "combined":
        dataset = CombinedDataset(
            dataset1=cbt_data,
            dataset2=easymath_data,
            tokenizer=tokenizer,
            max_length=block_size,
            balance=True,
        )
    elif args.dataset_mode == "combined_guidance":
        dataset = CombinedDatasetForGuidance(
            dataset1=cbt_data,
            dataset2=easymath_data,
            tokenizer=tokenizer,
            max_length=block_size,
            balance=True,
        )
    elif args.dataset_mode == "cbt":
        dataset = CBTDataset(
            dataset=cbt_data,
            tokenizer=tokenizer,
            max_length=block_size,
        )
    elif args.dataset_mode == "easymath":
        dataset = EasyMathDataset(
            dataset=easymath_data,
            tokenizer=tokenizer,
            max_length=block_size,
        )
    else:
        raise ValueError(f"Unsupported dataset_mode: {args.dataset_mode}")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader_dl = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )
    val_loader_dl = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    train_loader = course_data_generator(train_loader_dl, device)
    course_val_steps = len(val_loader_dl)
    course_seq_len = block_size

training_time_ms = 0
torch.cuda.synchronize()
t0 = time.perf_counter()

last_completed_step = start_step - 1
for step in range(start_step, args.num_iterations + 1):
    last_step = step == args.num_iterations

    # Validation
    if last_step or (
        args.val_loss_every > 0 and step % args.val_loss_every == 0
    ):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)

        # validation via evaluate()
        if use_fineweb:
            val_batch = world_size * args.val_seq_len
            assert args.val_tokens % val_batch == 0
            val_steps = args.val_tokens // val_batch
            val_loader = distributed_data_generator(args.val_files, val_batch, rank, world_size)
        else:
            val_loader = course_data_generator(val_loader_dl, device)
            val_steps = course_val_steps
        val_loss = evaluate(model, val_loader, val_steps)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        num_toks = step * args.grad_accum_steps_per_device \
            * (course_seq_len or args.train_seq_len) * world_size
        print0(
            f"step:{step}/{args.num_iterations} "
            f"val_loss:{val_loss:.4f} "
            f"train_time:{training_time_ms:.0f}ms "
            f"step_avg:{training_time_ms / max(step, 1):.2f}ms "
            f"tokens:{num_toks / 1e6:.2f}M",
            console=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        persist_checkpoint(step, final=False)
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if last_step and master_process and args.save_checkpoint:
            ckpt = dict(
                step=step,
                code=code,
                model=model.state_dict(),
                optimizers=[opt.state_dict() for opt in optimizers],
            )
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(ckpt, f"logs/{run_id}/state_step{step:06d}.pt")

        if last_step:
            break

    # Training
    train_step(model, train_loader, step, optimizers, optimizer2, args.grad_accum_steps_per_device)

    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step + 1}/{args.num_iterations} "
        f"train_time:{approx_time:.0f}ms "
        f"step_avg:{approx_time / (step + 1):.2f}ms",
        console=True,
    )
    last_completed_step = step

persist_checkpoint(last_completed_step, final=True)

print0(
    f"peak memory allocated: "
    f"{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: "
    f"{torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
    console=True,
)
dist.destroy_process_group()
