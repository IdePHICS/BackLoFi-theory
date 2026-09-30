"""Paper figure (appendix): test error against the number of passes of the
fixed-width backward correction on the fully connected network.

Standalone: reads only ``results/backward_rowcol_sweep/cifar10_ffn/layers_3``,
forward LoFi (``fwd``) and the 10-pass first-order arm ``rcfopcitm``
(column-only update, per-channel scale, importance swap, forward-only tau) at
r in {2, 3, 5, 7, 10, 12}% of k.  One row, one panel per training-set size; pass 0 is
forward LoFi; every point is the best filter width k for that (n, r, pass),
selected by the seed mean; bands are one standard error over seeds.  Same style
as ``plot_paper_test_error_vs_n.py`` (ICLR text width, 9 pt Times).  Writes
``passes_ffn.pdf`` and ``.png`` and prints the table.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_data import add_data_args, dump_cells, load_cells  # noqa: E402

DATA_DIR = Path("results/backward_rowcol_sweep/cifar10_ffn/layers_3")
OUT_DIR = Path("imgs/paper")
STEM = "passes_ffn"
TEXTWIDTH_IN = 5.5  # ICLR \textwidth
TAG = "rcfopcitm"
RX = re.compile(
    r"^animal_vehicle_seed\d+_ntrain(?P<n>\d+)_ntest\d+_L3_p5000spk(?P<k>\d+)relu_"
    rf"(?P<arm>fwd|{TAG}_r(?P<r>\d+)_pass(?P<p>\d+))\.json$"
)
R_STYLE = {  # Okabe–Ito
    2: dict(color="#56B4E9", marker="^"),
    3: dict(color="#009E73", marker="v"),
    5: dict(color="#E69F00", marker="s"),
    7: dict(color="#CC79A7", marker="D"),
    10: dict(color="#D55E00", marker="o"),
    12: dict(color="#000000", marker="P"),
}
FWD_COLOR = "#0072B2"


def load(d: Path, ns: list[int]) -> dict[tuple, list[float]]:
    """(n, k, r, pass) -> errors over seeds; forward LoFi is r = pass = 0."""
    cells: dict[tuple, list[float]] = defaultdict(list)
    for e in os.scandir(d):
        m = RX.match(e.name)
        if m is None or int(m.group("n")) not in ns:
            continue
        with open(e.path) as fh:
            acc = json.load(fh)["final"].get("test_accuracy")
        if acc is None:
            continue
        r, p = (0, 0) if m.group("arm") == "fwd" else (int(m["r"]), int(m["p"]))
        cells[(int(m["n"]), int(m["k"]), r, p)].append(100.0 * (1.0 - acc))
    return cells


def best(cells: dict, n: int, r: int, p: int) -> tuple[float, float, int] | None:
    """(mean, sem, k) of the best seed-mean filter width."""
    out = None
    for (nn, k, rr, pp), v in cells.items():
        if (nn, rr, pp) != (n, r, p) or len(v) < 2:
            continue
        m = statistics.mean(v)
        if out is None or m < out[0]:
            out = (m, statistics.stdev(v) / math.sqrt(len(v)), k)
    return out


def _nlab(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 else str(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ns", type=int, nargs="+", default=[1000, 5000, 10000, 50000])
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
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    data_f = args.data_dir / f"{STEM}.csv"
    if args.from_data:
        cells = load_cells(
            data_f,
            lambda n, k, arm, r, p: (
                (n, int(k), 0, 0) if arm == "fwd" else (n, int(k), int(r), int(p))
            ),
        )
        cells = {key: v for key, v in cells.items() if key[0] in args.ns}
    else:
        cells = load(DATA_DIR, args.ns)
        if args.dump_data:
            dump_cells(
                cells,
                data_f,
                lambda key: (
                    (key[0], str(key[1]), "fwd", "", "")
                    if key[2] == 0
                    else (key[0], str(key[1]), TAG, key[2], key[3])
                ),
            )
            print(f"wrote {data_f}")
    passes = sorted({p for (_n, _k, _r, p) in cells if p > 0})
    rs = sorted({r for (_n, _k, r, _p) in cells if r > 0})
    fig, axes = plt.subplots(
        1, len(args.ns), figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / 3.4), squeeze=False
    )
    for ax, n in zip(axes[0], args.ns, strict=True):
        fwd = best(cells, n, 0, 0)
        print(f"\nn = {n}: forward {fwd[0]:.2f} ± {fwd[1]:.2f} (k={fwd[2]})")
        ax.axhline(fwd[0], color=FWD_COLOR, lw=0.8, ls=":")
        for r in rs:
            pts = [(0, fwd)] + [(p, b) for p in passes if (b := best(cells, n, r, p))]
            x = [p for p, _ in pts]
            m = [b[0] for _, b in pts]
            s = [b[1] for _, b in pts]
            ax.fill_between(
                x,
                [a - b for a, b in zip(m, s, strict=True)],
                [a + b for a, b in zip(m, s, strict=True)],
                color=R_STYLE[r]["color"],
                alpha=0.2,
                lw=0,
            )
            ax.plot(x, m, lw=1.0, ms=2.2, label=f"$r = {r}\\%$", **R_STYLE[r])
            i = min(range(len(m)), key=m.__getitem__)
            print(
                f"  r={r:2d}%: "
                + " ".join(f"{v:.2f}" for v in m[1:])
                + f"   best {m[i]:.2f} ± {s[i]:.2f} at pass {x[i]} (k={pts[i][1][2]})"
            )
        ax.set_title(f"$n$ = {_nlab(n)}")
        ax.set_xlabel("Pass", labelpad=1)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("Test Error (%)")
    axes[0][0].legend(
        frameon=False,
        loc="upper left",
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.15,
        borderpad=0.15,
        borderaxespad=0.25,
    )
    fig.tight_layout(pad=0.15, w_pad=0.6)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"{STEM}.{ext}", dpi=300)
    plt.close(fig)
    print(f"\nwrote {args.out_dir / STEM}.pdf and .png")


if __name__ == "__main__":
    main()
