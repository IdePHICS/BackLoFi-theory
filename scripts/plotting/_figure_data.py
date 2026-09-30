"""Tidy-CSV export / import of the numbers behind each paper figure.

The plotters read the raw per-cell JSONs in ``results/`` (hundreds of
thousands of files).  With ``--dump-data`` they write the exact numbers they
use to ``data/<figure>.csv``; with ``--from-data`` they redraw from those files
without ``results/``.

Cells: one row per seed of one cell, columns ``n, width, arm, r, pass, idx,
test_error`` (``idx`` = position of the seed in the cell's list, ``r``/``pass``
empty for forward and baseline arms; test error in percent).  The plotters map
their own cell keys to and from these columns with ``to_row`` / ``from_row``.
GD curves: one row per recorded step, columns ``n, lr, seed, step,
test_error``.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

CELL_COLS = ["n", "width", "arm", "r", "pass", "idx", "test_error"]
CURVE_COLS = ["n", "lr", "seed", "step", "test_error"]


def dump_cells(
    cells: dict, path: Path, to_row: Callable[[tuple], tuple[int, str, str, str, str]]
) -> None:
    """Write ``{key: [errors]}``; ``to_row(key) -> (n, width, arm, r, pass)``."""
    rows = []
    for key, v in cells.items():
        n, width, arm, r, p = to_row(key)
        rows += [(n, width, arm, r, p, i, repr(e)) for i, e in enumerate(v)]
    rows.sort(key=lambda t: (t[0], str(t[1]), t[2], str(t[3]), str(t[4]), t[5]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CELL_COLS)
        w.writerows(rows)


def load_cells(
    path: Path, from_row: Callable[[int, str, str, str, str], tuple]
) -> dict:
    """Inverse of :func:`dump_cells`; ``from_row(n, width, arm, r, pass) -> key``."""
    cells: dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = from_row(
                int(row["n"]), row["width"], row["arm"], row["r"], row["pass"]
            )
            cells.setdefault(key, []).append(float(row["test_error"]))
    return cells


def dump_curves(curves: dict, path: Path) -> None:
    """Write ``{(n, lr, seed): {step: error}}``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CURVE_COLS)
        for (n, lr, s), c in sorted(curves.items()):
            for st in sorted(c):
                w.writerow((n, f"{lr:g}", s, st, repr(c[st])))


def load_curves(path: Path) -> dict:
    """Inverse of :func:`dump_curves`."""
    curves: dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["n"]), float(row["lr"]), int(row["seed"]))
            curves.setdefault(key, {})[int(row["step"])] = float(row["test_error"])
    return curves


def add_data_args(ap, default_dir: str = "data") -> None:
    ap.add_argument("--data-dir", type=Path, default=Path(default_dir))
    ap.add_argument(
        "--dump-data",
        action="store_true",
        help="write the figure's numbers to data-dir",
    )
    ap.add_argument(
        "--from-data",
        action="store_true",
        help="read them from data-dir instead of results/",
    )
