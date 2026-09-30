"""Are the passes of the row-then-column correction incremental?

Runs the first-order, column-only FFN arm for a few passes at one cell and
records the thin SVD of ``A_l`` at every (pass, boundary).  Reports, for every
pass, the overlap of its read (write) directions with those of the previous
pass and with the span of ALL earlier passes: the mean squared cosine
``||V_t^T Q||_F^2 / r`` (1 = inside the span, ``r / p`` for a random subspace).

Writes ``results/rowcol_pass_overlap/overlap_seed{s}.json`` and
``imgs/rowcol_small_r/pass_overlap.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path[:0] = ["src", "scripts"]
import neural_lofi.training.backward_rowcol as br  # noqa: E402

OUT = Path("results/rowcol_pass_overlap")
FIG = Path("imgs/rowcol_small_r/pass_overlap.png")


def _ov(a: torch.Tensor, q: torch.Tensor) -> float:
    return float((a.transpose(0, 1) @ q).pow(2).sum() / a.shape[1])


def run(r_frac: float, args: argparse.Namespace) -> dict:
    import sweep_backward_rowcol as sw

    rec: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    orig = br.RowColBackwardTrainer._thin_svd

    def spy(self, a, r):  # type: ignore[no-untyped-def]
        u, s, v = orig(self, a, r)
        rec.append((self._pass, u.double().cpu(), v.double().cpu()))
        return u, s, v

    br.RowColBackwardTrainer._thin_svd = spy  # type: ignore[method-assign]
    sys.argv = [
        "sweep",
        "--conf",
        "conf/backward_rowcol_fopcitm_ffn_3l.yaml",
        "--override",
        f"job=k{args.k}",
        f"dataset.n_train={args.n}",
        f"seed={args.seed}",
        f"device={args.device}",
        f"r_fracs=[{r_frac}]",
        f"n_passes={args.passes}",
        f"output_dir={OUT}/cells_r{round(100 * r_frac)}",
        f"ckpt_dir={OUT}/ckpt",
    ]
    try:
        sw.main()
    finally:
        br.RowColBackwardTrainer._thin_svd = orig  # type: ignore[method-assign]
    out: dict = {}
    for bi, name in ((0, "boundary 1"), (1, "boundary 0")):  # top-down order
        seq = [x for i, x in enumerate(rec) if i % 2 == bi]
        rows = []
        for t in range(1, len(seq)):
            qv, _ = torch.linalg.qr(torch.cat([s[2] for s in seq[:t]], dim=1))
            qu, _ = torch.linalg.qr(torch.cat([s[1] for s in seq[:t]], dim=1))
            rows.append(
                {
                    "pass": seq[t][0],
                    "read_prev": _ov(seq[t][2], seq[t - 1][2]),
                    "read_span": _ov(seq[t][2], qv),
                    "write_prev": _ov(seq[t][1], seq[t - 1][1]),
                    "write_span": _ov(seq[t][1], qu),
                }
            )
        v0 = seq[0][2]
        out[name] = {
            "rank": v0.shape[1],
            "random": v0.shape[1] / v0.shape[0],
            "rows": rows,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--r-fracs", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {f"{round(100 * r)}": run(r, args) for r in args.r_fracs}
    (OUT / f"overlap_seed{args.seed}.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    colors = {"2": "#1b9e77", "5": "#d95f02", "10": "#7570b3"}
    for ax, name in zip(axes, ("boundary 1", "boundary 0"), strict=True):
        for r, by in res.items():
            rows = by[name]["rows"]
            xs = [x["pass"] for x in rows]
            c = colors.get(r, "k")
            ax.plot(
                xs,
                [x["read_span"] for x in rows],
                "-o",
                color=c,
                ms=4,
                label=f"reads, r = {r}%",
            )
            ax.plot(
                xs,
                [x["write_span"] for x in rows],
                "--s",
                color=c,
                ms=4,
                label=f"writes, r = {r}%",
            )
            ax.axhline(by[name]["random"], color=c, lw=0.6, ls=":")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("pass")
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction inside the span of earlier passes")
    axes[1].legend(frameon=False, fontsize=7, ncol=1, loc="lower left")
    fig.suptitle(
        f"New directions of each pass vs those already installed (FFN, k={args.k},"
        f" n={args.n}); dotted: random subspace",
        fontsize=9,
    )
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    for r, by in res.items():
        for name, b in by.items():
            last = b["rows"][-1]
            print(
                f"r={r}% {name}: pass {last['pass']} reads in earlier span"
                f" {last['read_span']:.3f}, writes {last['write_span']:.3f}"
                f" (random {b['random']:.4f})"
            )
    print(f"wrote {OUT}/overlap_seed{args.seed}.json and {FIG}")


if __name__ == "__main__":
    main()
