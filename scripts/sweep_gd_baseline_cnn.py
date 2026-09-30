"""Full-batch gradient-descent baseline on the CNN of the row-then-column
campaign: the protocol of the FFN baseline (``sweep_gd_baseline.py``).

One invocation = one ``(lr, n_train, seed)`` cell.  The model is the folded GD
twin of the block config (:meth:`BackpropModel.from_block_config`): conv 4096,
conv 4096 + max-pool, FC 5000 and a linear scalar readout, the architecture the
LoFi network realises with low-rank weights.  Training is plain constant-step
gradient descent (no momentum, no weight decay) on the mean squared error
against the plus/minus-one labels, with the **whole training set per step**:
the exact gradient is accumulated over chunks of ``chunk_size`` images, each
chunk contributing ``sum((f - y)^2) / n``, then one optimizer step is taken.

Test error is the fraction of wrong signs on the held-out set, recorded at
log-spaced step counts up to ``max_steps``; the training MSE of every step is
the exact full-batch loss before that step.  One JSON per cell::

    {stem}_lr{lr}.json   with "curve": [{"step", "train_mse", "test_error"}, ...]

with ``stem = {preset}_seed{s}_ntrain{n}_ntest{t}_L3_{arch}gd``.  The curve is
flushed to ``.partial.json`` while the run is alive and the full state to
``.ckpt.pt`` every ``ckpt_every_minutes`` and at the end, from which an
interrupted cell resumes and a finished cell is EXTENDED when ``max_steps`` is
raised (the checkpoint steps are fixed by ``checkpoint_steps`` so that curves
of different runs share their steps).

Under ``torchrun`` (``WORLD_SIZE`` > 1) the step is data-parallel and still the
exact full-batch one: rank ``r`` accumulates the gradient of its share
``X[r::world]`` with the same ``1 / n`` normalisation, the gradients are summed
over ranks (``all_reduce``) and every rank takes the same step from the same
seeded initialisation.  Rank 0 alone evaluates, logs and writes.
"""

import contextlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.distributed as dist
from _run_utils import save_results
from config_utils import parse_classes, with_config
from omegaconf import OmegaConf

from neural_lofi.datasets import load_tensors
from neural_lofi.models.backprop import BackpropModel
from neural_lofi.models.block_config import parse_block_config
from neural_lofi.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)


def _checkpoints(max_steps: int, n_points: int) -> list[int]:
    """Log-spaced step counts in ``[1, max_steps]``, deduplicated."""
    pts = {
        int(round(10 ** (i * math.log10(max_steps) / (n_points - 1))))
        for i in range(n_points)
    }
    return sorted(s for s in pts if 1 <= s <= max_steps)


@with_config("conf/gd_baseline_cnn.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    d = cfg.dataset
    lr = float(cfg.lr)
    stem = (
        f"{d.preset}_seed{cfg.seed}_ntrain{d.n_train}_ntest{d.n_test}"
        f"_L{len(cfg.blocks)}_{cfg.arch_tag}gd"
    )
    path = out_dir / f"{stem}_lr{lr:g}.json"
    ckpt = path.with_suffix(".ckpt.pt")
    if path.exists() and not force_run:
        done = int(json.loads(path.read_text())["gd"]["max_steps"])
        if done >= int(cfg.max_steps):
            log.info("[lr=%g n=%d seed=%d] exists.", lr, d.n_train, cfg.seed)
            return
        log.info(
            "[lr=%g n=%d seed=%d] extending %d -> %d steps (%s)",
            lr,
            d.n_train,
            cfg.seed,
            done,
            int(cfg.max_steps),
            "from its checkpoint" if ckpt.exists() else "no checkpoint: from scratch",
        )

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    set_seed(cfg.seed)
    if world > 1:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        device = f"cuda:{local}"
    else:
        device = resolve_device(cfg.device)
    common: dict[str, Any] = dict(
        dataset_type=d.name,
        device=device,
        seed=cfg.seed,
        root=cfg.root,
        class_preset=d.get("preset"),
        classes=parse_classes(cfg),
        remap_labels=cfg.remap_labels,
        flatten=False,
    )
    X, y = load_tensors(
        split=str(d.get("train_split", "train")), n_samples=d.n_train, **common
    )
    Xte, yte = load_tensors(split="test", n_samples=d.n_test, **common)
    dev = torch.device(device)
    bf16 = str(cfg.precision) == "bf16"
    fmt = torch.channels_last if bf16 else torch.contiguous_format
    X = X.to(dev).contiguous(memory_format=fmt)
    Xte = Xte.to(dev).contiguous(memory_format=fmt)
    y, yte = y.to(dev).float().reshape(-1), yte.to(dev).float().reshape(-1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    def _amp() -> contextlib.AbstractContextManager:
        if bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    set_seed(cfg.seed)
    model = (
        BackpropModel.from_block_config(
            parse_block_config(
                {"blocks": OmegaConf.to_object(cfg.blocks), "final_reduce": {"k": None}}
            ),
            input_shape=tuple(X.shape[1:]),
            output_dim=1,
        )
        .to(dev)
        .to(memory_format=fmt)
    )
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)

    @torch.no_grad()
    def _test_error() -> float:
        model.eval()
        wrong = 0
        for i in range(0, Xte.shape[0], 256):
            with _amp():
                pred = model(Xte[i : i + 256]).reshape(-1).float()
            wrong += int((torch.sign(pred) != yte[i : i + 256]).sum().item())
        model.train()
        return wrong / Xte.shape[0]

    n, chunk = int(X.shape[0]), int(cfg.chunk_size)  # n: the FULL batch
    max_steps = int(cfg.max_steps)
    if cfg.get("checkpoint_steps"):
        marks = {int(m) for m in cfg.checkpoint_steps if int(m) <= max_steps}
    else:
        marks = set(_checkpoints(max_steps, int(cfg.n_checkpoints)))
    marks.add(max_steps)
    var_y = float(y.var(unbiased=False))
    Xs, ys = X[rank::world], y[rank::world]  # this rank's share of the batch
    partial = path.with_suffix(".partial.json")
    first, minutes0, t0, last_save = 1, 0.0, time.time(), time.time()
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        curve, first, minutes0 = st["curve"], int(st["step"]) + 1, st["minutes"]
        log.info("[lr=%g n=%d seed=%d] resumed at step %d", lr, n, cfg.seed, first)
    else:
        te0 = _test_error() if rank == 0 else float("nan")
        curve = [{"step": 0, "train_mse": float("nan"), "test_error": te0}]

    def _minutes() -> float:
        return minutes0 + (time.time() - t0) / 60

    def _save_ckpt(step: int) -> None:
        tmp = ckpt.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "curve": curve,
                "step": step,
                "minutes": _minutes(),
            },
            tmp,
        )
        tmp.replace(ckpt)  # kept after the run: a raised max_steps resumes here

    diverged = False
    for step in range(first, max_steps + 1):
        opt.zero_grad(set_to_none=True)
        mse = 0.0
        for i in range(0, Xs.shape[0], chunk):
            with _amp():
                out = model(Xs[i : i + chunk]).reshape(-1).float()
            loss = (out - ys[i : i + chunk]).pow(2).sum() / n
            loss.backward()  # gradients accumulate: exactly the full-batch one
            mse += float(loss.detach())
        if world > 1:  # sum the shares: the same full-batch gradient on every rank
            total = torch.tensor(mse, dtype=torch.float64, device=dev)
            dist.all_reduce(total)
            mse = float(total)
            for prm in model.parameters():
                dist.all_reduce(prm.grad)
        opt.step()
        diverged = not math.isfinite(mse) or mse > 100.0 * var_y
        if rank == 0 and (step in marks or diverged):
            te = float("nan") if diverged else _test_error()
            curve.append({"step": step, "train_mse": mse, "test_error": te})
            log.info(
                "[lr=%g n=%d seed=%d] step %d: train_mse=%.4f test_err=%.4f (%.0f min)",
                lr,
                n,
                cfg.seed,
                step,
                mse,
                te,
                _minutes(),
            )
            save_results(partial, {"curve": curve, "reason": "running"})
        if diverged:
            break
        due = time.time() - last_save > 60.0 * float(cfg.ckpt_every_minutes)
        if rank == 0 and (due or step == max_steps):
            _save_ckpt(step)
            last_save = time.time()

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return
    ok = [c for c in curve if math.isfinite(c["test_error"])]
    best = min(ok, key=lambda c: c["test_error"])
    save_results(
        path,
        {
            "curve": curve,
            "final": {
                "test_error": curve[-1]["test_error"],
                "best_test_error": best["test_error"],
                "best_step": best["step"],
                "diverged": diverged,
                "minutes": _minutes(),
            },
            "gd": {
                "lr": lr,
                "max_steps": max_steps,
                "optimizer": "sgd",
                "full_batch": True,
                "world_size": world,
                "chunk_size": chunk,
                "precision": str(cfg.precision),
                "n_params": n_params,
            },
        },
    )
    partial.unlink(missing_ok=True)
    log.info(
        "[lr=%g n=%d seed=%d] best test_err=%.4f at step %d (final %.4f)",
        lr,
        n,
        cfg.seed,
        best["test_error"],
        best["step"],
        curve[-1]["test_error"],
    )


if __name__ == "__main__":
    main()
