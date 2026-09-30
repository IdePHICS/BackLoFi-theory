"""Local launcher of the full-batch CNN GD baseline on a 2-GPU workstation.

Runs ``scripts/sweep_gd_baseline_cnn.py`` cells two at a time, one per GPU, in
order of increasing ``n`` (3000 full-batch steps cost ``3000 x n x 5.8 ms``:
2.4 h at n = 500, doubling with n) and, within an ``n``, of decreasing learning
rate.  Cells are skip-if-exists and resume from their checkpoint, so the
launcher can be stopped and restarted at any time.  Logs go to
``results/gd_baseline_cnn/logs``; progress to ``launcher.log`` there.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path("results/gd_baseline_cnn/cifar10_cnn/layers_3")
LOGS = Path("results/gd_baseline_cnn/logs")
STEM = "animal_vehicle_seed{s}_ntrain{n}_ntest10000_L3_c4096p5000mpgd_lr{lr:g}.json"
_lock = threading.Lock()


def say(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M')} {msg}"
    print(line, flush=True)
    with _lock, open(LOGS / "launcher.log", "a") as fh:
        fh.write(line + "\n")


def result(lr: float, n: int, s: int) -> dict | None:
    p = OUT / STEM.format(s=s, n=n, lr=lr)
    return json.loads(p.read_text())["final"] if p.exists() else None


def run_cells(cells: list[tuple[float, int, int]], gpus: list[int]) -> None:
    """Run ``(lr, n, seed)`` cells in order, one at a time per GPU."""
    queue = list(cells)

    def worker(gpu: int) -> None:
        while True:
            with _lock:
                if not queue:
                    return
                lr, n, s = queue.pop(0)
            if result(lr, n, s) is None:
                say(f"gpu{gpu}: start lr={lr:g} n={n} seed={s}")
                with open(LOGS / f"lr{lr:g}_n{n}_seed{s}.log", "a") as fh:
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/sweep_gd_baseline_cnn.py",
                            "--conf",
                            "conf/gd_baseline_cnn.yaml",
                            "--override",
                            f"lr={lr}",
                            f"dataset.n_train={n}",
                            f"seed={s}",
                            f"device=cuda:{gpu}",
                        ],
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
            r = result(lr, n, s)
            say(
                f"gpu{gpu}: lr={lr:g} n={n} seed={s} -> "
                + (
                    ("DIVERGED" if r["diverged"] else "done")
                    + f", best {100 * r['best_test_error']:.2f}% at step"
                    f" {r['best_step']}, final {100 * r['test_error']:.2f}%"
                    f" ({r['minutes']:.0f} min)"
                    if r
                    else "NO RESULT (see the cell log)"
                )
            )

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-3, 3e-3, 1e-3, 3e-4])
    ap.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 5000, 10000, 20000, 50000],
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    say(f"launcher start: lrs={args.lrs} ns={args.ns} seeds={args.seeds}")
    lrs = sorted(args.lrs, reverse=True)
    run_cells(
        [(lr, n, s) for n in sorted(args.ns) for s in args.seeds for lr in lrs],
        args.gpus,
    )
    say("ALL DONE")


if __name__ == "__main__":
    main()
