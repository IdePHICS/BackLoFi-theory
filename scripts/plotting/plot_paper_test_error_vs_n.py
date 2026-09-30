"""Paper figures: test error against n on binary CIFAR-10, one file per network.

Standalone: reads only the series it draws and nothing else.

- ``results/backward_rowcol_sweep/cifar10_ffn/baselines``: linear ridge on the
  pixels (FFN figure only) and the 3-layer random-features network.
- ``results/backward_rowcol_sweep/cifar10_ffn/layers_3``: forward LoFi (``fwd``)
  and the backward arms ``rcfopcit`` (first-order chain) and ``rcexpcit`` (exact
  chain), both with per-channel scale, importance swap and forward-only tau.
- ``results/gd_baseline_3k/cifar10_ffn/layers_3``: full-batch GD on the folded
  FFN twin (FFN figure only): common best learning rate, best of 3000 steps.
- ``results/backward_rowcol_cnn_sweep/cifar10_cnn/baselines``: random features
  with the CNN architecture (max-pool).
- ``results/backward_rowcol_cnn_sweep/cifar10_cnn/layers_3``: forward LoFi and
  the arms ``rcfopci`` (first-order chain) and ``rcexpci`` (exact chain).

Per n the forward curve is the best filter width and each backward curve the best
(width, r, pass); "best" is by the seed mean, shaded bands are one standard
error over seeds.  Each figure is half the ICLR text width (0.48 x 5.5 in) and
set in 9 pt Times (Nimbus Roman, the font of the ``times`` package).  Writes
``test_error_vs_n_ffn`` and ``test_error_vs_n_cnn`` as ``.pdf`` and ``.png``
and prints the selection tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_data import (  # noqa: E402
    add_data_args,
    dump_cells,
    dump_curves,
    load_cells,
    load_curves,
)
from _gd_utils import gd_selection, load_gd_curves  # noqa: E402

FFN_DIR = Path("results/backward_rowcol_sweep/cifar10_ffn")
CNN_DIR = Path("results/backward_rowcol_cnn_sweep/cifar10_cnn")
GD_DIR = Path("results/gd_baseline_3k/cifar10_ffn/layers_3")
GD_LRS = [0.01, 0.03, 0.1, 0.2]
GD_SEEDS = [0, 1, 2, 3, 4]
GD_CNN_DIR = Path("results/gd_baseline_cnn/cifar10_cnn/layers_3")
GD_CNN_LRS = [5e-3, 3e-3, 1e-3]
GD_CNN_SEEDS = [0, 1, 2]
OUT_DIR = Path("imgs/paper")
STEM = "test_error_vs_n"
N_TRAINS = [500, 1000, 2000, 5000, 10000, 20000, 50000]
TEXTWIDTH_IN = 5.5  # ICLR \textwidth
FIG_W = 0.48 * TEXTWIDTH_IN
FIG_H = FIG_W / 1.3

_PRE = r"^animal_vehicle_seed(?P<seed>\d+)_ntrain(?P<n>\d+)_ntest\d+_"
FFN_RE = re.compile(
    _PRE + r"L3_p5000spk(?P<k>\d+)relu_"
    r"(?P<arm>fwd|(?P<tag>rcfopcit|rcexpcit)_r(?P<r>\d+)_pass(?P<p>\d+))\.json$"
)
CNN_RE = re.compile(
    _PRE + r"L3_c4096p5000mpkc(?P<kc>\d+)k3(?P<k3>\d+)relu_"
    r"(?P<arm>fwd|(?P<tag>rcfopci|rcexpci)_r(?P<r>\d+)_pass(?P<p>\d+))\.json$"
)
LIN_RE = re.compile(_PRE + r"D3072_linear\.json$")
RF_FFN_RE = re.compile(_PRE + r"L3_p5000rfk0relu_rf\.json$")
RF_CNN_RE = re.compile(_PRE + r"L3_c4096p5000mpk0relu_rf\.json$")

# label, series key, style — the same on both figures (Okabe–Ito colours);
# "gd": FFN at the common best rate; CNN at the best rate run at each n.
SERIES = [
    ("Ridge", "linear", dict(color="#8c8c8c", ls=":", marker="o")),
    ("RF", "rf", dict(color="#E69F00", ls="--", marker="^")),
    ("FwdLoFi", "fwd", dict(color="#0072B2", ls="-", marker="s")),
    (
        "BackLoFi ($1^{\\mathrm{st}}$ ord)",
        "fo",
        dict(color="#D55E00", ls="-", marker="o"),
    ),
    ("BackLoFi (exact)", "ex", dict(color="#009E73", ls="-.", marker="D")),
    ("GD", "gd", dict(color="#000000", ls=(0, (4, 1.5)), marker="x")),
]
Y_TICKS = [4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40]
# Legend placement (two columns) and the y limits that leave its band free:
# the FFN legend sits above the flat ridge curve (top raised), the CNN legend
# above the random-features curve.  None: 0.93 x the smallest / 1.07 x the
# largest mean.
LEGEND = {
    "ffn": dict(loc="upper right", ncol=2),
    "cnn": dict(loc="upper right", ncol=2),
}
YLIM = {"ffn": (None, 27.0), "cnn": (None, None)}

Cells = dict[tuple, list[float]]  # (n, width, arm) -> errors over seeds


def _err(path: str) -> float | None:
    with open(path) as fh:
        acc = json.load(fh)["final"].get("test_accuracy")
    return None if acc is None else 100.0 * (1.0 - float(acc))


def scan(d: Path, rx: re.Pattern, arm_of, width_of) -> Cells:
    """One os.scandir pass; keeps only files matching ``rx`` at the listed n."""
    cells: Cells = defaultdict(list)
    for e in os.scandir(d):
        m = rx.match(e.name)
        if m is None or int(m.group("n")) not in N_TRAINS:
            continue
        v = _err(e.path)
        if v is not None:
            cells[(int(m.group("n")), width_of(m), arm_of(m))].append(v)
    return cells


def _none(m: re.Match) -> None:
    return None


def _arm(m: re.Match):
    if m.group("arm") == "fwd":
        return "fwd"
    return (m.group("tag"), int(m.group("r")), int(m.group("p")))


def load_panel(kind: str) -> Cells:
    cells: Cells = defaultdict(list)
    if kind == "ffn":
        cells.update(
            scan(FFN_DIR / "layers_3", FFN_RE, _arm, lambda m: int(m.group("k")))
        )
        cells.update(scan(FFN_DIR / "baselines", RF_FFN_RE, lambda m: "rf", _none))
    else:
        cells.update(
            scan(
                CNN_DIR / "layers_3",
                CNN_RE,
                _arm,
                lambda m: (int(m.group("kc")), int(m.group("k3"))),
            )
        )
        cells.update(scan(CNN_DIR / "baselines", RF_CNN_RE, lambda m: "rf", _none))
        return cells
    cells.update(scan(FFN_DIR / "baselines", LIN_RE, lambda m: "linear", _none))
    return cells


def _keep(key: str):
    tags = {"fo": ("rcfopcit", "rcfopci"), "ex": ("rcexpcit", "rcexpci")}
    if key in tags:
        return lambda arm: isinstance(arm, tuple) and arm[0] in tags[key]
    return lambda arm: arm == key


def select(cells: Cells, n: int, keep, allowed: set | None = None):
    """Best seed-mean over (width, arm) at n: (mean, sem, width, arm, n_seeds)."""
    best = None
    for (nn, w, arm), v in cells.items():
        if nn != n or not keep(arm) or len(v) < 2:
            continue
        if allowed is not None and (w, arm[1], arm[2]) not in allowed:
            continue
        m = statistics.mean(v)
        if best is None or m < best[0]:
            best = (m, statistics.stdev(v) / math.sqrt(len(v)), w, arm, len(v))
    return best


def curves(cells: Cells) -> dict[str, list[tuple]]:
    """key -> [(n, mean, sem, width, arm, n_seeds)] for the five series."""
    out: dict[str, list[tuple]] = {}
    for _, key, _ in SERIES:
        if key == "gd":
            continue
        pts = []
        for n in N_TRAINS:
            b = select(cells, n, _keep(key))
            if b is not None:
                pts.append((n, *b))
        out[key] = pts
    # Fairness check (printed only): first-order restricted to the exact arm's
    # own (width, r, pass) grid, which is a subset of the first-order grid.
    pts = []
    for n in N_TRAINS:
        grid = {
            (w, arm[1], arm[2])
            for (nn, w, arm), v in cells.items()
            if nn == n and _keep("ex")(arm) and len(v) >= 2
        }
        b = select(cells, n, _keep("fo"), allowed=grid) if grid else None
        if b is not None:
            pts.append((n, *b))
    out["fo_on_ex_grid"] = pts
    return out


def gd_curve(kind: str = "ffn", raw: dict | None = None) -> list[tuple]:
    """Full-batch GD, best of the logged steps, in the
    ``(n, mean, sem, width, arm, n_seeds)`` layout.  FFN: the common best
    learning rate (every rate was run at every n).  CNN: each n uses the best
    of the rates run there with every seed (seed-mean selection; 1e-3 has a
    single seed at n >= 5k and is not a candidate there)."""
    if kind == "ffn":
        if raw is None:
            raw = load_gd_curves(GD_DIR, N_TRAINS, GD_SEEDS)
        if not raw:
            return []
        sel = gd_selection(raw, N_TRAINS, GD_LRS, GD_SEEDS)
        lr = sel["common_lr"]
        return [
            (n, m, s, None, ("gd", lr, step), len(GD_SEEDS))
            for n in N_TRAINS
            if n in sel["common"]
            for (m, s, step) in [sel["common"][n]]
        ]
    if raw is None:
        raw = load_gd_curves(GD_CNN_DIR, N_TRAINS, GD_CNN_SEEDS)
    if not raw:
        return []
    sel = gd_selection(
        raw, N_TRAINS, GD_CNN_LRS, GD_CNN_SEEDS, min_seeds=len(GD_CNN_SEEDS)
    )
    return [
        (n, m, s, None, ("gd", lr, step), len(GD_CNN_SEEDS))
        for n in N_TRAINS
        if n in sel["per_n_best"]
        for (m, s, step, lr) in [sel["per_n_best"][n]]
    ]


def _nlab(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 else str(n)


def _fmt(pt: tuple) -> str:
    n, m, s, w, arm, ns = pt
    if w is None:
        hp = ""
    elif isinstance(w, int):
        hp = f"k={w}"
    else:
        hp = f"kc={w[0]},k3={w[1]}"
    if isinstance(arm, tuple) and arm[0] == "gd":
        hp = f"lr={arm[1]},step={arm[2]}"
    elif isinstance(arm, tuple):
        hp += f",r={arm[1]},pass={arm[2]}"
    return f"{m:6.2f} ± {s:4.2f} [{hp}; {ns} seeds]"


def print_table(name: str, cv: dict[str, list[tuple]]) -> None:
    print(f"\n== {name} ==")
    extra = ("first-order on the exact arm's grid", "fo_on_ex_grid", None)
    for label, key, _ in [*SERIES, extra]:
        if not cv.get(key):
            continue
        print(f"  {label}")
        for pt in cv[key]:
            print(f"    n={_nlab(pt[0]):>4}: {_fmt(pt)}")


def draw(cv: dict[str, list[tuple]], title: str, out: Path, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for label, key, st in SERIES:
        pts = cv.get(key, [])
        if len(pts) < 2:
            continue
        x = [p[0] for p in pts]
        m = [p[1] for p in pts]
        s = [p[2] for p in pts]
        ax.fill_between(
            x,
            [a - b for a, b in zip(m, s, strict=True)],
            [a + b for a, b in zip(m, s, strict=True)],
            color=st["color"],
            alpha=0.2,
            lw=0,
        )
        ax.plot(x, m, label=label, lw=1.0, ms=2.2, **st)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ys = [p[1] for _, key, _ in SERIES for p in cv.get(key, [])]
    lo = YLIM[kind][0] or 0.93 * min(ys)
    hi = YLIM[kind][1] or 1.07 * max(ys)
    ticks = [t for t in Y_TICKS if lo <= t <= hi]
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("$n$", labelpad=1)
    ax.set_ylabel("Test Error (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(
        frameon=False,
        fontsize=6.5,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.1,
        columnspacing=0.6,
        borderpad=0.15,
        borderaxespad=0.25,
        **LEGEND[kind],
    )
    fig.tight_layout(pad=0.15)
    stem = f"{STEM}_{kind}"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {out / stem}.pdf and .png")


# --- tidy-CSV form of a cell key (n, width, arm): width '' | '512' | '512x128';
# arm 'fwd' | 'rf' | 'linear' | tag with r and pass.
def _to_row(key: tuple) -> tuple:
    n, w, arm = key
    ws = "" if w is None else (f"{w[0]}x{w[1]}" if isinstance(w, tuple) else str(w))
    if isinstance(arm, tuple):
        return (n, ws, arm[0], arm[1], arm[2])
    return (n, ws, arm, "", "")


def _from_row(n: int, ws: str, arm: str, r: str, p: str) -> tuple:
    w = (
        None
        if ws == ""
        else (tuple(int(t) for t in ws.split("x")) if "x" in ws else int(ws))
    )
    return (n, w, (arm, int(r), int(p)) if r != "" else arm)


def _data(kind: str, args) -> tuple[Cells, dict]:
    """Cells and raw GD curves of one panel, from results/ or from data/."""
    cells_f = args.data_dir / f"{STEM}_{kind}.csv"
    gd_f = args.data_dir / f"gd_curves_{kind}.csv"
    if args.from_data:
        return load_cells(cells_f, _from_row), load_curves(gd_f)
    cells = load_panel(kind)
    if kind == "ffn":
        raw = load_gd_curves(GD_DIR, N_TRAINS, GD_SEEDS)
    else:
        raw = load_gd_curves(GD_CNN_DIR, N_TRAINS, GD_CNN_SEEDS)
    if args.dump_data:
        dump_cells(cells, cells_f, _to_row)
        dump_curves(raw, gd_f)
        print(f"wrote {cells_f} and {gd_f}")
    return cells, raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    add_data_args(ap)
    args = ap.parse_args()
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "STIXGeneral", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells, raw = _data("ffn", args)
    ffn = curves(cells)
    ffn["gd"] = gd_curve("ffn", raw)
    cells, raw = _data("cnn", args)
    cnn = curves(cells)
    cnn["gd"] = gd_curve("cnn", raw)
    print_table("Fully connected network", ffn)
    print_table("Convolutional network", cnn)
    draw(ffn, "Fully connected network", args.out_dir, "ffn")
    draw(cnn, "Convolutional network", args.out_dir, "cnn")


if __name__ == "__main__":
    main()
