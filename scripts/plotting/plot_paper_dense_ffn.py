"""Paper figure (appendix): test error against the number of passes of the
damped dense update ``W <- (1 - alpha) W + alpha * estimate`` on the fully
connected network, next to the fixed-width swap of the main figure.

Standalone: reads only ``results/backward_rowcol_sweep/cifar10_ffn/layers_3``:
forward LoFi (``fwd``), the damped arm ``rcfodia10`` (alpha = 0.1, 50 passes,
r in {5, 10, 25}% of k) and the 4-pass swap arm ``rcfopcit`` of the main
figure.  One row, one panel per training-set size; pass 0 is forward LoFi;
every point is the best filter width k for that (n, r, pass), selected by the
seed mean; bands are one standard error over seeds.  The swap arm is drawn at
its best r for that n.  Same style as ``plot_paper_passes_ffn.py``.  Writes
``dense_ffn.pdf`` and ``.png`` and prints the table.
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
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_data import add_data_args, dump_cells, load_cells  # noqa: E402

DATA_DIR = Path("results/backward_rowcol_sweep/cifar10_ffn/layers_3")
OUT_DIR = Path("imgs/paper")
STEM = "dense_ffn"
TEXTWIDTH_IN = 5.5  # ICLR \textwidth
DAMPED, SWAP = "rcfodia10", "rcfopcit"
RX = re.compile(
    r"^animal_vehicle_seed\d+_ntrain(?P<n>\d+)_ntest\d+_L3_p5000spk(?P<k>\d+)relu_"
    rf"(?P<arm>fwd|(?P<tag>{DAMPED}|{SWAP})_r(?P<r>\d+)_pass(?P<p>\d+))\.json$"
)
R_STYLE = {  # Okabe–Ito
    5: dict(color="#56B4E9", marker="^"),
    10: dict(color="#E69F00", marker="s"),
    25: dict(color="#D55E00", marker="o"),
}
SWAP_STYLE = dict(color="#009E73", marker="D", ls="--")
FWD_COLOR = "#0072B2"


def load(d: Path, ns: list[int]) -> dict[tuple, list[float]]:
    """(n, k, tag, r, pass) -> errors over seeds; forward is ('fwd', 0, 0)."""
    cells: dict[tuple, list[float]] = defaultdict(list)
    for e in os.scandir(d):
        m = RX.match(e.name)
        if m is None or int(m.group("n")) not in ns:
            continue
        with open(e.path) as fh:
            acc = json.load(fh)["final"].get("test_accuracy")
        if acc is None:
            continue
        if m.group("arm") == "fwd":
            key = (int(m["n"]), int(m["k"]), "fwd", 0, 0)
        else:
            key = (int(m["n"]), int(m["k"]), m["tag"], int(m["r"]), int(m["p"]))
        cells[key].append(100.0 * (1.0 - acc))
    return cells


def best(cells: dict, n: int, tag: str, r: int, p: int) -> tuple | None:
    """(mean, sem, k) of the best seed-mean filter width."""
    out = None
    for (nn, k, tt, rr, pp), v in cells.items():
        if (nn, tt, rr, pp) != (n, tag, r, p) or len(v) < 2:
            continue
        m = statistics.mean(v)
        if out is None or m < out[0]:
            out = (m, statistics.stdev(v) / math.sqrt(len(v)), k)
    return out


def _nlab(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 else str(n)


def _band(ax, x, pts, **st) -> None:
    m = [b[0] for b in pts]
    s = [b[1] for b in pts]
    ax.fill_between(
        x,
        [a - b for a, b in zip(m, s, strict=True)],
        [a + b for a, b in zip(m, s, strict=True)],
        color=st["color"],
        alpha=0.2,
        lw=0,
    )
    ax.plot(x, m, lw=1.0, ms=2.0, **st)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--ns", type=int, nargs="+", default=[500, 1000, 5000, 10000, 50000]
    )
    ap.add_argument("--rs", type=int, nargs="+", default=[10, 25])
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
            "legend.fontsize": 6.5,
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
                (n, int(k), "fwd", 0, 0)
                if arm == "fwd"
                else (n, int(k), arm, int(r), int(p))
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
                    if key[2] == "fwd"
                    else (key[0], str(key[1]), key[2], key[3], key[4])
                ),
            )
            print(f"wrote {data_f}")
    passes = sorted({p for (_n, _k, t, _r, p) in cells if t == DAMPED})
    fig, axes = plt.subplots(
        1, len(args.ns), figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / 3.2), squeeze=False
    )
    for ax, n in zip(axes[0], args.ns, strict=True):
        fwd = best(cells, n, "fwd", 0, 0)
        print(f"\nn = {n}: forward {fwd[0]:.2f} ± {fwd[1]:.2f} (k={fwd[2]})")
        ax.axhline(fwd[0], color=FWD_COLOR, lw=0.8, ls=":", label="FwdLoFi")
        for r in args.rs:
            pts = [(0, fwd)] + [
                (p, b) for p in passes if (b := best(cells, n, DAMPED, r, p))
            ]
            x = [p for p, _ in pts]
            _band(
                ax, x, [b for _, b in pts], label=f"damped, $r = {r}\\%$", **R_STYLE[r]
            )
            m = [b[0] for _, b in pts]
            i = min(range(len(m)), key=m.__getitem__)
            at = dict(zip(x, m, strict=True))
            print(
                f"  damped r={r:2d}%: "
                + " ".join(f"p{p}={at[p]:.2f}" for p in (1, 10, 25, 50) if p in at)
                + f"   best {m[i]:.2f} ± {pts[i][1][1]:.2f} at pass {x[i]}"
                f" (k={pts[i][1][2]})"
            )
        # the swap arm of the main figure at its best (r, pass) for this n
        sw = [
            (b, r, p)
            for (nn, _k, t, r, p) in cells
            if nn == n and t == SWAP
            for b in [best(cells, n, SWAP, r, p)]
            if b is not None
        ]
        if sw:
            r_b = min(sw, key=lambda t: t[0][0])[1]
            sp = sorted(
                {p for (nn, _k, t, r, p) in cells if (nn, t, r) == (n, SWAP, r_b)}
            )
            pts = [(0, fwd)] + [(p, best(cells, n, SWAP, r_b, p)) for p in sp]
            _band(
                ax,
                [p for p, _ in pts],
                [b for _, b in pts],
                label="swap (best $r$)",
                **SWAP_STYLE,
            )
            bm = min(pts[1:], key=lambda t: t[1][0])
            print(
                f"  swap r={r_b:2d}%: "
                + " ".join(f"p{p}={b[0]:.2f}" for p, b in pts[1:])
                + f"   best {bm[1][0]:.2f} ± {bm[1][1]:.2f} at pass {bm[0]}"
                f" (k={bm[1][2]})"
            )
        ax.set_title(f"$n$ = {_nlab(n)}")
        ax.set_xlabel("Pass", labelpad=1)
        ax.set_xticks([0, 25, 50])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("Test Error (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.0),
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.15, w_pad=0.5, rect=(0, 0, 1, 0.9))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"{STEM}.{ext}", dpi=300)
    plt.close(fig)
    print(f"\nwrote {args.out_dir / STEM}.pdf and .png")


if __name__ == "__main__":
    main()
