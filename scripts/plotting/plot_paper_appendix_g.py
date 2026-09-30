"""Paper appendix "Additional numerical simulations": the pass-overlap figure
and the gradient-descent training curves (the number-of-passes and damped-update
figures have their own scripts, ``plot_paper_passes_ffn.py`` and
``plot_paper_dense_ffn.py``).

- ``pass_overlap.{pdf,png}``: fraction of each pass's new read / write directions
  inside the span of those installed by earlier passes
  (``results/rowcol_pass_overlap/overlap_seed*.json``, from
  ``scripts/probe_rowcol_pass_overlap.py``).
- ``gd_curves_{ffn,cnn}.{pdf,png}``: test error of full-batch GD against the step
  for each learning rate, at four n, with the BackLoFi level of the main figure.

Every number is a seed mean with its sem; every selection is by the seed mean.
Same style as ``plot_paper_test_error_vs_n.py`` (ICLR text width, 9 pt Times).
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
from _figure_data import add_data_args, load_cells, load_curves  # noqa: E402
from _gd_utils import load_gd_curves, mean_curve  # noqa: E402

FFN_DIR = Path("results/backward_rowcol_sweep/cifar10_ffn/layers_3")
CNN_DIR = Path("results/backward_rowcol_cnn_sweep/cifar10_cnn/layers_3")
GD_DIR = Path("results/gd_baseline_3k/cifar10_ffn/layers_3")
GD_CNN_DIR = Path("results/gd_baseline_cnn/cifar10_cnn/layers_3")
OVERLAP_DIR = Path("results/rowcol_pass_overlap")
OUT_DIR = Path("imgs/paper")
N_TRAINS = [500, 1000, 2000, 5000, 10000, 20000, 50000]
TEXTWIDTH_IN = 5.5
BLUE, RED, GREEN, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9"
_PRE = r"^animal_vehicle_seed\d+_ntrain(?P<n>\d+)_ntest\d+_"
_ARM = r"(?P<arm>fwd|(?P<tag>rc[a-z]*)_r(?P<r>\d+)_pass(?P<p>\d+))\.json$"
FFN_RE = re.compile(_PRE + r"L3_p5000spk(?P<k>\d+)relu_" + _ARM)
CNN_RE = re.compile(_PRE + r"L3_c4096p5000mpkc(?P<kc>\d+)k3(?P<k3>\d+)relu_" + _ARM)
FFN_TAGS = {"rcfopcit"}

Cells = dict[tuple, list[float]]  # (n, width, arm) -> errors; arm 'fwd' | (tag, r, p)


def load(d: Path, rx: re.Pattern, tags: set[str]) -> Cells:
    cells: Cells = defaultdict(list)
    for e in os.scandir(d):
        m = rx.match(e.name)
        if m is None or (m["tag"] is not None and m["tag"] not in tags):
            continue
        with open(e.path) as fh:
            acc = json.load(fh)["final"].get("test_accuracy")
        if acc is None:
            continue
        gd = m.groupdict()
        w = int(gd["k"]) if gd.get("k") else (int(gd["kc"]), int(gd["k3"]))
        arm = "fwd" if gd["arm"] == "fwd" else (gd["tag"], int(gd["r"]), int(gd["p"]))
        cells[(int(gd["n"]), w, arm)].append(100.0 * (1.0 - acc))
    return cells


def _ms(v: list[float]) -> tuple[float, float]:
    return statistics.mean(v), statistics.stdev(v) / math.sqrt(len(v))


def best(cells: Cells, n: int, keep) -> tuple | None:
    out = None
    for (nn, w, arm), v in cells.items():
        if nn == n and keep(w, arm) and len(v) >= 2:
            m, s = _ms(v)
            if out is None or m < out[0]:
                out = (m, s, w, arm)
    return out


def _style() -> None:
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
            "xtick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _band(ax, x, m, s, **st) -> None:
    ax.fill_between(
        x,
        [a - b for a, b in zip(m, s, strict=True)],
        [a + b for a, b in zip(m, s, strict=True)],
        color=st["color"],
        alpha=0.18,
        lw=0,
    )
    ax.plot(x, m, lw=1.0, ms=2.2, **st)


def _legend(ax, **kw) -> None:
    ax.legend(
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        labelspacing=0.15,
        borderpad=0.15,
        borderaxespad=0.25,
        **kw,
    )


def _save(fig, out: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {out / stem}.pdf and .png")


def _nlab(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 else str(n)


# --------------------------------------------------------------------------- #


def fig_overlap(out: Path, overlap_dir: Path = OVERLAP_DIR) -> None:
    files = sorted(overlap_dir.glob("overlap_seed*.json"))
    if not files:
        print("no overlap files; skipped")
        return
    runs = [json.loads(f.read_text()) for f in files]
    fig, axes = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / 3.0), sharey=True
    )
    colors = {"2": SKY, "5": ORANGE, "10": RED}
    titles = {"boundary 1": "Boundary 2 | 3", "boundary 0": "Boundary 1 | 2"}
    print(f"\n== pass overlap: {len(runs)} seeds ==")
    for ax, name in zip(axes, ("boundary 1", "boundary 0"), strict=True):
        for r, c in colors.items():
            rows = [run[r][name]["rows"] for run in runs if r in run]
            if not rows:
                continue
            xs = [x["pass"] for x in rows[0]]
            for key, ls, mk, lab in (
                ("read_span", "-", "o", "read"),
                ("write_span", "--", "s", "write"),
            ):
                vals = [[row[i][key] for row in rows] for i in range(len(xs))]
                m = [statistics.mean(v) for v in vals]
                s = [
                    statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                    for v in vals
                ]
                _band(
                    ax, xs, m, s, color=c, ls=ls, marker=mk, label=f"{lab}, $r={r}\\%$"
                )
                print(f"  {name} r={r}% {lab}: " + " ".join(f"{v:.3f}" for v in m))
        ax.set_title(titles[name])
        ax.set_xlabel("Pass", labelpad=1)
        ax.grid(alpha=0.25, lw=0.5)
    lowest = min(min(ln.get_ydata()) for ax in axes for ln in ax.get_lines())
    axes[0].set_ylim(math.floor(10 * (lowest - 0.05)) / 10, 1.02)
    axes[0].set_ylabel("Fraction in span")
    _legend(axes[0], loc="lower left", ncol=2, columnspacing=0.8)
    fig.tight_layout(pad=0.15, w_pad=0.8)
    fig.subplots_adjust(left=0.09)  # tight_layout clips the shared-axis y label
    _save(fig, out, "pass_overlap")


def fig_gd_curves(
    cells: Cells,
    out: Path,
    ns: list[int],
    *,
    gd_dir: Path,
    lrs: list[float],
    seeds: list[int],
    tag: str,
    stem: str,
    min_seeds: int,
    gd_curves: dict | None = None,
    top_factor: float = 1.8,
) -> None:
    """Test error of full-batch GD against the step, one panel per n, one curve
    per learning rate (largest in black), BackLoFi ``tag`` as the dotted level."""
    curves = load_gd_curves(gd_dir, ns, seeds) if gd_curves is None else gd_curves
    fig, axes = plt.subplots(
        1, len(ns), figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / 3.4), squeeze=False
    )
    small = [(SKY, "^"), (ORANGE, "s"), (GREEN, "D")][: len(lrs) - 1]
    lr_style = dict(zip(sorted(lrs), [*small, ("#000000", "x")], strict=True))
    print(f"\n== GD curves ({stem}) ==")
    for ax, n in zip(axes[0], ns, strict=True):
        back = best(cells, n, lambda w, a: isinstance(a, tuple) and a[0] == tag)
        ax.axhline(back[0], color=RED, lw=0.9, ls=":", label="BackLoFi")
        top, lows = top_factor * back[0], [back[0]]
        for lr in sorted(lrs):
            mc = mean_curve(curves, n, lr, seeds, min_seeds)
            if not mc:
                continue
            steps = [s for s in sorted(mc) if s > 0]
            lows += [mc[s][0] for s in steps if mc[s][0] <= top]
            c, mk = lr_style[lr]
            _band(
                ax,
                steps,
                [mc[s][0] for s in steps],
                [mc[s][1] for s in steps],
                color=c,
                marker=mk,
                ls="-",
                label=f"GD, lr $={lr:g}$",
            )
            k = sum((n, lr, sd) in curves for sd in seeds)
            print(
                f"  n={n:6d} lr={lr:g}: {k} seeds, last step {steps[-1]},"
                f" error {mc[steps[-1]][0]:.2f}"
            )
        ax.set_xscale("log")
        ax.set_xlim(5, 4000)
        ax.set_title(f"$n$ = {_nlab(n)}")
        ax.set_xlabel("Step", labelpad=1)
        ax.set_ylim(0.93 * min(lows), top)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("Test Error (%)")
    by_label = {}
    for ax in axes[0]:  # a rate may be missing at some n: collect over the panels
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            by_label.setdefault(lab, h)
    order = ["BackLoFi"] + [f"GD, lr $={lr:g}$" for lr in sorted(lrs)]
    labels = [lab for lab in order if lab in by_label]
    fig.legend(
        [by_label[lab] for lab in labels],
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.15, w_pad=0.6, rect=(0, 0, 1, 0.9))
    _save(fig, out, stem)


def _from_row(n: int, ws: str, arm: str, r: str, p: str) -> tuple:
    """Cell key of the main figure's data files (see plot_paper_test_error_vs_n)."""
    w = (
        None
        if ws == ""
        else (tuple(int(t) for t in ws.split("x")) if "x" in ws else int(ws))
    )
    return (n, w, (arm, int(r), int(p)) if r != "" else arm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    add_data_args(ap)
    ap.add_argument("--gd-ns", type=int, nargs="+", default=[1000, 5000, 10000, 50000])
    ap.add_argument(
        "--gd-cnn-ns", type=int, nargs="+", default=[1000, 5000, 10000, 20000]
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    if args.from_data:
        ffn = load_cells(args.data_dir / "test_error_vs_n_ffn.csv", _from_row)
        cnn = load_cells(args.data_dir / "test_error_vs_n_cnn.csv", _from_row)
        gd_ffn = load_curves(args.data_dir / "gd_curves_ffn.csv")
        gd_cnn = load_curves(args.data_dir / "gd_curves_cnn.csv")
        overlap_dir = args.data_dir / "pass_overlap"
    else:
        ffn = load(FFN_DIR, FFN_RE, FFN_TAGS)
        cnn = load(CNN_DIR, CNN_RE, {"rcfopci"})
        gd_ffn = gd_cnn = None
        overlap_dir = OVERLAP_DIR
        if args.dump_data:  # the pass-overlap runs are copied as they are
            d = args.data_dir / "pass_overlap"
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted(OVERLAP_DIR.glob("overlap_seed*.json")):
                (d / f.name).write_bytes(f.read_bytes())
            print(f"wrote {d}/overlap_seed*.json")
    fig_overlap(args.out_dir, overlap_dir)
    fig_gd_curves(
        ffn,
        args.out_dir,
        args.gd_ns,
        gd_dir=GD_DIR,
        lrs=[0.01, 0.03, 0.1],
        seeds=[0, 1, 2, 3, 4],
        tag="rcfopcit",
        stem="gd_curves_ffn",
        min_seeds=2,
        gd_curves=gd_ffn,
    )
    fig_gd_curves(
        cnn,
        args.out_dir,
        args.gd_cnn_ns,
        gd_dir=GD_CNN_DIR,
        lrs=[1e-3, 3e-3, 5e-3],
        seeds=[0, 1, 2],
        tag="rcfopci",
        stem="gd_curves_cnn",
        min_seeds=1,
        gd_curves=gd_cnn,
        top_factor=3.0,
    )


if __name__ == "__main__":
    main()
