"""Row-then-column backward sweep on the 2-conv + avg-pool + FC network.

One invocation = one ``job: kc<val>_k3<val>`` for one ``(n_train, seed)``.
The job sets conv2's channel reduce to ``k_c`` and the FC's to ``k_3``,
obtains the **forward source model** once — loaded from ``ckpt_dir`` when a
checkpoint exists, otherwise fitted and checkpointed — and then, for every
``r`` in ``cfg.r_fracs``, runs
:class:`~neural_lofi.training.backward_rowcol_cnn.RowColCNNTrainer` for
``cfg.n_passes`` sequential passes from that frozen model, saving after every
pass.

Files in ``output_dir``::

    {stem}_fwd.json                   forward LoFi at (k_c, k_3)
    {stem}_{tag}_r{pct}_pass{p}.json  row-then-column, rank fraction pct%, pass p

with ``stem = {preset}_seed{s}_ntrain{n}_ntest{t}_L{L}_{arch}kc{k_c}k3{k_3}{act}``
and ``{ckpt_dir}/{stem}_fwd.pt`` the forward state dict.
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
from neural_lofi.training.backward_rowcol import RowColConfig
from neural_lofi.training.backward_rowcol_cnn import RowColCNNTrainer
from neural_lofi.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)

_JOB_RE = re.compile(r"^kc(\d+)_k3(\d+)$")


def _stem(cfg: Any, k_c: int, k_3: int) -> str:
    d = cfg.dataset
    arch = cfg.get("arch_tag") or f"c{cfg.conv_p}p{cfg.fc_p}"
    return (
        f"{d.preset}_seed{cfg.seed}_ntrain{d.n_train}_ntest{d.n_test}"
        f"_L{len(cfg.blocks)}_{arch}kc{k_c}k3{k_3}{cfg.activation}"
    )


@with_config("conf/backward_rowcol_cnn_first_order.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    m = _JOB_RE.match(str(cfg.job))
    if m is None:
        raise ValueError(f"job must be kc<val>_k3<val>, got {cfg.job!r}")
    k_c, k_3 = int(m.group(1)), int(m.group(2))
    cfg.k_c, cfg.k_3 = k_c, k_3
    cfg.blocks[1].reduce.k = k_c
    cfg.blocks[2].reduce.k = k_3
    stem = _stem(cfg, k_c, k_3)
    r_fracs = [float(r) for r in cfg.r_fracs]
    n_passes = int(cfg.n_passes)
    ckpt_dir = Path(cfg.ckpt_dir) if cfg.get("ckpt_dir") else None

    fwd_path = out_dir / f"{stem}_fwd.json"
    ckpt_path = ckpt_dir / f"{stem}_fwd.pt" if ckpt_dir is not None else None
    tag = str(cfg.get("arm_tag", "rcmfpci"))

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
        flatten=False,
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
        "[%s] n_train=%d seed=%d: r todo=%s",
        cfg.job,
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
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        log.info("[%s] forward loaded from %s", cfg.job, ckpt_path)
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
            "[%s] forward kc=%d k3=%d: test_acc=%s",
            cfg.job,
            k_c,
            k_3,
            fwd_results["final"].get("test_accuracy"),
        )

    for r in todo:

        def _save(idx: int, results: dict[str, Any], _r: float = r) -> None:
            path = arm_path(_r, idx)
            if force_run or not path.exists():
                save_results(path, results)
            log.info(
                "[%s] rowcol r=%.2f pass=%d: test_acc=%s origins=%s",
                cfg.job,
                _r,
                idx,
                results["final"].get("test_accuracy"),
                results["backward"]["origins"],
            )

        trainer = RowColCNNTrainer(
            copy.deepcopy(model),
            RowColConfig(
                r_frac=r,
                beta=float(cfg.beta),
                n_passes=n_passes,
                chain=str(cfg.get("chain", "mean_field")),
                column_scale=str(cfg.get("column_scale", "per_channel")),
                swap_rule=str(cfg.get("swap_rule", "importance")),
                chunk_size=int(cfg.get("chunk_size", 128)),
                lift_seed=int(cfg.seed),
                tau_over=str(cfg.get("tau_over", "all")),
                upper_update=str(cfg.get("upper_update", "column")),
                **ridge,
            ),
        )
        trainer.fit(train_loader, test_loader=test_loader, pass_callback=_save)


if __name__ == "__main__":
    main()
