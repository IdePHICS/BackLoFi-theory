"""Loading and selection helpers for the full-batch GD baseline curves.

One JSON per ``(n, lr, seed)`` cell (``scripts/sweep_gd_baseline.py``) holds
``curve = [{"step", "train_mse", "test_error"}, ...]``.  Every selection here
is made on the SEED-MEAN curve (never a per-seed oracle): the error of a cell
at a step is the mean over seeds, the best step of ``(n, lr)`` minimizes that
mean, and the common learning rate minimizes the mean over ``n`` of the
best-step errors.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from pathlib import Path

GD_RE = re.compile(
    r"^animal_vehicle_seed(?P<seed>\d+)_ntrain(?P<ntrain>\d+)_ntest\d+"
    r"_L(?P<L>\d)_(?P<arch>[a-z0-9]+?)gd_lr(?P<lr>[0-9.e+-]+)\.json$"
)


def load_gd_curves(
    d: Path, ntrains: list[int], seeds: list[int]
) -> dict[tuple[int, float, int], dict[int, float]]:
    """``(n, lr, seed) -> {step: test error %}`` for every JSON in ``d``."""
    out: dict[tuple[int, float, int], dict[int, float]] = {}
    if not d.is_dir():
        return out
    for e in os.scandir(d):
        m = GD_RE.match(e.name)
        if m is None:
            continue
        n, s = int(m.group("ntrain")), int(m.group("seed"))
        if n not in ntrains or s not in seeds:
            continue
        with open(e.path) as fh:
            curve = json.load(fh)["curve"]
        out[(n, float(m.group("lr")), s)] = {
            int(c["step"]): 100.0 * float(c["test_error"])
            for c in curve
            if c["test_error"] is not None and math.isfinite(float(c["test_error"]))
        }
    return out


def mean_curve(
    curves: dict, n: int, lr: float, seeds: list[int], min_seeds: int = 2
) -> dict[int, tuple[float, float]] | None:
    """Seed-mean curve ``{step: (mean, sem)}`` over the steps present in every
    seed; ``None`` with fewer than ``min_seeds`` seeds (sem 0 for one seed)."""
    cs = [curves[(n, lr, s)] for s in seeds if (n, lr, s) in curves]
    if len(cs) < min_seeds:
        return None
    steps = sorted(set.intersection(*(set(c) for c in cs)))
    out = {}
    for st in steps:
        v = [c[st] for c in cs]
        sem = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
        out[st] = (statistics.mean(v), sem)
    return out


def best_step(
    curves: dict, n: int, lr: float, seeds: list[int], min_seeds: int = 2
) -> tuple[float, float, int] | None:
    """``(mean, sem, step)`` of the best seed-mean error of ``(n, lr)``."""
    mc = mean_curve(curves, n, lr, seeds, min_seeds)
    if not mc:
        return None
    st = min(mc, key=lambda k: mc[k][0])
    return (mc[st][0], mc[st][1], st)


def gd_selection(
    curves: dict,
    ntrains: list[int],
    lrs: list[float],
    seeds: list[int],
    min_seeds: int = 2,
) -> dict:
    """Common learning rate and per-``n`` results.

    Returns ``{"common_lr": lr, "common": {n: (mean, sem, step)},
    "per_n_best": {n: (mean, sem, step, lr)}, "table": {(n, lr): (...)}}``;
    the common lr minimizes the mean over the ``n`` that have every lr."""
    table = {
        (n, lr): b
        for n in ntrains
        for lr in lrs
        if (b := best_step(curves, n, lr, seeds, min_seeds))
    }
    full_n = [n for n in ntrains if all((n, lr) in table for lr in lrs)]
    common_lr = None
    if full_n:
        common_lr = min(
            lrs, key=lambda lr: statistics.mean(table[(n, lr)][0] for n in full_n)
        )
    per_n_best = {}
    for n in ntrains:
        cands = [(table[(n, lr)], lr) for lr in lrs if (n, lr) in table]
        if cands:
            b, lr = min(cands, key=lambda t: t[0][0])
            per_n_best[n] = (*b, lr)
    common = {
        n: table[(n, common_lr)]
        for n in ntrains
        if common_lr is not None and (n, common_lr) in table
    }
    return {
        "common_lr": common_lr,
        "common": common,
        "per_n_best": per_n_best,
        "table": table,
    }
