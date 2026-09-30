"""Row-then-column backward sweep (the backward step of the paper).

One invocation = one ``job: k<val>`` for one ``(n_train, seed)``.  The job sets
every reduce of the base config to ``k``, obtains the **forward source model**
once — loaded from ``ckpt_dir`` when a checkpoint exists, otherwise fitted and
checkpointed — and then, for every ``r`` in ``cfg.r_fracs``, runs
:class:`~neural_lofi.training.backward_rowcol.RowColBackwardTrainer` for
``cfg.n_passes`` sequential passes from that frozen model, saving after every
pass.  The forward model is therefore fitted at most once per ``(k, n, seed)``
across every backward arm, now and later.

Files in ``output_dir``::

    {stem}_fwd.json                   forward LoFi at k (first-moment column on)
    {stem}_{tag}_r{pct}_pass{p}.json  row-then-column, rank r = pct% of k, pass p
                                      (tag = cfg.arm_tag: rc local, rcmf mean-field)

with ``stem = {preset}_seed{s}_ntrain{n}_ntest{t}_L{L}_{arch}k{k}{act}``, and
``{ckpt_dir}/{stem}_fwd.pt`` the forward state dict.
"""

import copy
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _block_model import build_block_model
from _run_utils import save_results
from config_utils import parse_classes, with_config
from torch.utils.data import DataLoader

from neural_lofi.datasets import load_tensors
from neural_lofi.training import SpectralTrainer, SpectralTrainerConfig
from neural_lofi.training.backward_rowcol import RowColBackwardTrainer, RowColConfig
from neural_lofi.training.backward_rowcol_dense import (
    RowColDenseConfig,
    RowColDenseTrainer,
)
from neural_lofi.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)

_JOB_RE = re.compile(r"^k(\d+)$")


def _stem(cfg: Any, k: int) -> str:
    d = cfg.dataset
    arch = cfg.get("arch_tag") or f"p{cfg.all_p}"
    return (
        f"{d.preset}_seed{cfg.seed}_ntrain{d.n_train}_ntest{d.n_test}"
        f"_L{len(cfg.blocks)}_{arch}k{k}{cfg.activation}"
    )


@with_config("conf/backward_rowcol_fopcit_ffn_3l.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    m = _JOB_RE.match(str(cfg.job))
    if m is None:
        raise ValueError(f"job must be k<val>, got {cfg.job!r}")
    k = int(m.group(1))
    cfg.all_k = k
    for b in cfg.blocks:
        if b.reduce.k is not None:
            b.reduce.k = k
    stem = _stem(cfg, k)
    r_fracs = [float(r) for r in cfg.r_fracs]
    n_passes = int(cfg.n_passes)
    ckpt_dir = Path(cfg.ckpt_dir) if cfg.get("ckpt_dir") else None

    fwd_path = out_dir / f"{stem}_fwd.json"
    ckpt_path = ckpt_dir / f"{stem}_fwd.pt" if ckpt_dir is not None else None

    tag = str(cfg.get("arm_tag", "rc"))
    if str(cfg.get("representation", "factored")) == "dense":
        tag += f"a{round(100 * float(cfg.alpha))}"  # alpha as a percent in the name

    def arm_path(r: float, p: int) -> Path:
        return out_dir / f"{stem}_{tag}_r{round(100 * r)}_pass{p}.json"

    todo = [
        r
        for r in r_fracs
        if force_run or any(not arm_path(r, p).exists() for p in range(1, n_passes + 1))
    ]
    if not todo and fwd_path.exists() and not force_run:
        log.info("[%s] all arms exist — nothing to do.", cfg.job)
        return

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    common: dict[str, Any] = dict(
        dataset_type=cfg.dataset.name,
        device=device,
        seed=cfg.seed,
        root=cfg.root,
        class_preset=cfg.dataset.get("preset"),
        classes=parse_classes(cfg),
        remap_labels=cfg.remap_labels,
        flatten=True,
    )
    X_train, y_train = load_tensors(
        split=str(cfg.dataset.get("train_split", "train")),
        n_samples=cfg.dataset.n_train,
        **common,
    )
    X_test, y_test = load_tensors(split="test", n_samples=cfg.dataset.n_test, **common)
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=cfg.batch_size)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=cfg.batch_size)
    log.info(
        "[%s] L=%d n_train=%d seed=%d: r todo=%s",
        cfg.job,
        len(cfg.blocks),
        cfg.dataset.n_train,
        cfg.seed,
        todo,
    )
    ridge = dict(
        device=device,
        verbose=False,
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        alpha_num=cfg.alpha_num,
    )

    # Forward source model: from the checkpoint when it exists, else fit + save.
    set_seed(cfg.seed)
    model = build_block_model(cfg, input_shape=tuple(X_train.shape[1:]), seed=cfg.seed)
    if (
        ckpt_path is not None
        and ckpt_path.exists()
        and fwd_path.exists()
        and not force_run
    ):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
        log.info("[%s] forward k=%d loaded from %s", cfg.job, k, ckpt_path)
    else:
        model, fwd_results = SpectralTrainer(model, SpectralTrainerConfig(**ridge)).fit(
            train_loader, test_loader=test_loader
        )
        if force_run or not fwd_path.exists():
            save_results(fwd_path, fwd_results)
        if ckpt_path is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            torch.save(model.state_dict(), ckpt_path)
        log.info(
            "[%s] forward k=%d: test_acc=%s",
            cfg.job,
            k,
            fwd_results["final"].get("test_accuracy"),
        )

    for r in todo:

        def _save(idx: int, results: dict[str, Any], _r: float = r) -> None:
            path = arm_path(_r, idx)
            if force_run or not path.exists():
                save_results(path, results)
            log.info(
                "[%s] rowcol r=%.2f pass=%d: test_acc=%s %s",
                cfg.job,
                _r,
                idx,
                results["final"].get("test_accuracy"),
                results["backward"].get("origins", results["backward"].get("rescaled")),
            )

        if str(cfg.get("representation", "factored")) == "dense":
            # Damped interpolation on dense layers (backward_rowcol_dense.py):
            # W <- (1 - alpha) W + alpha * estimate, no swap, no importance.
            trainer: Any = RowColDenseTrainer(
                copy.deepcopy(model),
                RowColDenseConfig(
                    r_frac=r,
                    beta=float(cfg.beta),
                    alpha=float(cfg.alpha),
                    n_passes=n_passes,
                    chain=str(cfg.get("chain", "first_order")),
                    chunk_size=int(cfg.get("chunk_size", 8192)),
                    lift_seed=int(cfg.seed),
                    drift_guard=float(cfg.get("drift_guard", 8.0)),
                    **ridge,
                ),
            )
        else:
            trainer = RowColBackwardTrainer(
                copy.deepcopy(model),
                RowColConfig(
                    r_frac=r,
                    beta=float(cfg.beta),
                    n_passes=n_passes,
                    chain=str(cfg.get("chain", "local")),
                    column_scale=str(cfg.get("column_scale", "max")),
                    swap_rule=str(cfg.get("swap_rule", "replace_own")),
                    chunk_size=int(cfg.get("chunk_size", 8192)),
                    lift_seed=int(cfg.seed),
                    tau_over=str(cfg.get("tau_over", "all")),
                    upper_update=str(cfg.get("upper_update", "column")),
                    **ridge,
                ),
            )
        trainer.fit(train_loader, test_loader=test_loader, pass_callback=_save)


if __name__ == "__main__":
    main()
