"""Full-batch gradient-descent baseline on the same architecture as Neural LoFi.

One invocation = one ``(lr, n_train, seed)`` cell.  The model is the folded GD
twin of the block config (:meth:`BackpropModel.from_block_config`): one
trainable dense layer of width ``expand.p`` per block plus a linear scalar
readout, i.e. the same architecture the LoFi network realises with low-rank
weights.  Training is plain constant-step gradient descent on the mean squared
error against the centred plus/minus-one labels, with the **whole training set
as one batch**, so one optimizer step is one pass over the data.

Test error is the fraction of wrong signs on the held-out set, exactly how
every LoFi arm is scored, recorded at log-spaced step counts.  One JSON per
cell holds the whole curve::

    {stem}_gd_lr{lr}.json   with "curve": [{"step", "train_mse", "test_error"}, ...]

with ``stem = {preset}_seed{s}_ntrain{n}_ntest{t}_L{L}_p{p}gd``.
"""

import logging
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _run_utils import save_results
from config_utils import parse_classes, with_config
from omegaconf import OmegaConf

from neural_lofi.datasets import load_tensors
from neural_lofi.models.backprop import BackpropModel
from neural_lofi.models.block_config import parse_block_config
from neural_lofi.training import BackpropConfig, BackpropTrainer
from neural_lofi.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)


def _checkpoints(max_steps: int, n_points: int) -> list[int]:
    """Log-spaced step counts in ``[1, max_steps]``, deduplicated."""
    pts = {
        int(round(10 ** (i * math.log10(max_steps) / (n_points - 1))))
        for i in range(n_points)
    }
    return sorted(s for s in pts if 1 <= s <= max_steps)


@with_config("conf/gd_baseline_3k_ffn_3l.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    d = cfg.dataset
    arch = cfg.get("arch_tag") or f"p{cfg.all_p}"
    lr = float(cfg.lr)
    stem = (
        f"{d.preset}_seed{cfg.seed}_ntrain{d.n_train}_ntest{d.n_test}"
        f"_L{len(cfg.blocks)}_{arch}gd"
    )
    path = out_dir / f"{stem}_lr{lr:g}.json"
    if path.exists() and not force_run:
        log.info(
            "[lr=%g n=%d seed=%d] exists — nothing to do.", lr, d.n_train, cfg.seed
        )
        return

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    common: dict[str, Any] = dict(
        dataset_type=d.name,
        device=device,
        seed=cfg.seed,
        root=cfg.root,
        class_preset=d.get("preset"),
        classes=parse_classes(cfg),
        remap_labels=cfg.remap_labels,
        flatten=True,
    )
    X, y = load_tensors(
        split=str(d.get("train_split", "train")), n_samples=d.n_train, **common
    )
    Xte, yte = load_tensors(split="test", n_samples=d.n_test, **common)
    dev = torch.device(device)
    X, y = X.to(dev), y.to(dev).float().reshape(-1)
    Xte, yte = Xte.to(dev), yte.to(dev).float().reshape(-1)

    set_seed(cfg.seed)
    model = BackpropModel.from_block_config(
        parse_block_config(
            {
                "blocks": OmegaConf.to_object(cfg.blocks),
                "final_reduce": {"k": None},
            }
        ),
        input_shape=tuple(X.shape[1:]),
        output_dim=1,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    max_steps = int(cfg.max_steps)
    marks = set(_checkpoints(max_steps, int(cfg.n_checkpoints)))
    curve: list[dict[str, float]] = []

    @torch.no_grad()
    def _test_error() -> float:
        model.eval()
        wrong = 0
        for i in range(0, Xte.shape[0], 8192):
            pred = model(Xte[i : i + 8192]).reshape(-1)
            wrong += int((torch.sign(pred) != yte[i : i + 8192]).sum().item())
        model.train()
        return wrong / Xte.shape[0]

    def _on_epoch(_trainer, epoch: int, entry: dict[str, Any]) -> None:
        step = epoch + 1
        loss = float(entry.get("loss", float("nan")))
        if step in marks or not math.isfinite(loss):
            curve.append({"step": step, "train_mse": loss, "test_error": _test_error()})
            log.info(
                "[lr=%g n=%d seed=%d] step %d: train_mse=%.4f test_err=%.4f",
                lr,
                d.n_train,
                cfg.seed,
                step,
                loss,
                curve[-1]["test_error"],
            )

    trainer = BackpropTrainer(
        model,
        BackpropConfig(
            device=device,
            verbose=False,
            epochs=max_steps,
            lr=lr,
            optimizer_cls=torch.optim.SGD,
            mode="end_to_end",
            weight_decay=0.0,
        ),
        loss_fn=torch.nn.functional.mse_loss,
    )
    curve.append({"step": 0, "train_mse": float("nan"), "test_error": _test_error()})
    trainer.fit([(X, y)], epoch_callbacks=[_on_epoch])

    finite = [c for c in curve if math.isfinite(c["train_mse"]) or c["step"] == 0]
    best = min(finite, key=lambda c: c["test_error"]) if finite else curve[-1]
    save_results(
        path,
        {
            "curve": curve,
            "final": {
                "test_error": curve[-1]["test_error"],
                "best_test_error": best["test_error"],
                "best_step": best["step"],
                "diverged": not math.isfinite(curve[-1]["train_mse"]),
            },
            "gd": {
                "lr": lr,
                "max_steps": max_steps,
                "optimizer": "sgd",
                "full_batch": True,
                "n_params": n_params,
                "width": int(cfg.all_p),
                "depth": len(cfg.blocks),
            },
        },
    )
    log.info(
        "[lr=%g n=%d seed=%d] best test_err=%.4f at step %d (final %.4f)",
        lr,
        d.n_train,
        cfg.seed,
        best["test_error"],
        best["step"],
        curve[-1]["test_error"],
    )


if __name__ == "__main__":
    main()
