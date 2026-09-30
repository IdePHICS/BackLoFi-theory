# Reproducing the real-data figures

All experiments: binary CIFAR-10 (vehicles +1, animals -1; the `animal_vehicle`
preset), a training sub-sample of size `n`, the 10,000 test images, ReLU,
ridge readout with leave-one-out penalty among 50 log-spaced values in
[1e-6, 1e6], test error = fraction of wrong signs. Every cell is one JSON in
`results/` named by its (seed, n, architecture, arm, r, pass); the plotting
scripts average over seeds and select configurations on the seed mean.

Commands are run from the repository root with the package installed
(`pip install -e .`). `{N}`, `{K}`, `{S}` etc. denote loop variables; the grids
are listed with each arm. `scripts/parallel_run.py` expands a sweep YAML into
the Cartesian product of its lists and runs `--n_workers` cells in parallel
(under `mpirun` it also splits the product across ranks). Cells are
skip-if-exists, so any sweep can be interrupted and relaunched.

## Fully connected network (test error against n, left panel; passes; pass overlap; damped update; GD training curves)

Architecture: three layers of width p = 5,000 with a filter of rank k at every
layer (`conf/backward_rowcol_*_ffn_3l.yaml`). Forward Neural LoFi is fitted
once per (k, n, seed) and checkpointed (`ckpt_dir`); each backward arm reuses
it and writes the forward cell (`_fwd.json`) plus one cell per (r, pass).
Grids: k in {32, 64, 128, 256, 512, 1024, 2048} (one sweep file per k),
n in {500, 1000, 2000, 5000, 10000, 20000, 50000}, seeds 0-9. CPU; a
(k, n) job with 10 seeds takes minutes (n = 500) to a few hours (n = 50,000,
k = 2,048) on a 72-core node with 10 workers.

**BackLoFi, first-order signal, fixed width** (`rcfopcit`; r/k in
{5, 10, 15, 20, 25} %, 4 passes, beta = 0.25; the test-error-against-n figure):

```bash
for N in 500 1000 2000 5000 10000 20000 50000; do
for K in 32 64 128 256 512 1024 2048; do
python scripts/parallel_run.py --script scripts/sweep_backward_rowcol.py \
  --conf conf/backward_rowcol_fopcit_ffn_3l.yaml \
  --sweep conf/sweeps/merged_jobs_k${K}_10seeds.yaml --n_workers 10 \
  --override dataset.n_train=${N} beta=0.25 \
             ckpt_dir=results/backward_rowcol_sweep/checkpoints
done; done
```

**BackLoFi, exact signal** (`rcexpcit`; r/k in {10, 20, 25} %, 4 passes;
k in {64, ..., 2048}): the same loops with
`--conf conf/backward_rowcol_expcit_ffn_3l.yaml`.

**Number of passes at small r** (`rcfopcitm`, the number-of-passes figure; 10 passes;
r/k in {2, 5, 10} % from the config, {3, 7} % and {12} % as overrides):

```bash
for N in 500 1000 2000 5000 10000 20000 50000; do
for K in 32 64 128 256 512 1024 2048; do
for RF in "" '"r_fracs=[0.03,0.07]"' '"r_fracs=[0.12]"'; do
python scripts/parallel_run.py --script scripts/sweep_backward_rowcol.py \
  --conf conf/backward_rowcol_fopcitm_ffn_3l.yaml \
  --sweep conf/sweeps/merged_jobs_k${K}_10seeds.yaml --n_workers 10 \
  --override dataset.n_train=${N} beta=0.25 ${RF} \
             ckpt_dir=results/backward_rowcol_sweep/checkpoints
done; done; done
```

**Damped dense update** (`rcfodia10`, the damped-update figure; alpha = 0.1, r/k in
{5, 10, 25} %, 50 passes; n in {500, 1000, 5000, 10000, 50000}):

```bash
for N in 500 1000 5000 10000 50000; do
for K in 32 64 128 256 512 1024 2048; do
python scripts/parallel_run.py --script scripts/sweep_backward_rowcol.py \
  --conf conf/backward_rowcol_fodi_ffn_3l.yaml \
  --sweep conf/sweeps/merged_jobs_k${K}_10seeds.yaml --n_workers 10 \
  --override dataset.n_train=${N} beta=0.25 alpha=0.1 n_passes=50 \
             ckpt_dir=results/backward_rowcol_sweep/checkpoints
done; done
```

**Ridge on the pixels and the random-features network** (identity filters,
10 seeds):

```bash
for N in 500 1000 2000 5000 10000 20000 50000; do
python scripts/parallel_run.py --script scripts/sweep_baselines_pixels_rf.py \
  --conf conf/baselines_pixels_rf.yaml --sweep conf/sweeps/seeds_10.yaml \
  --n_workers 10 --override dataset.n_train=${N}
done
```

**Full-batch gradient descent** (test-error and training-curve figures; trainable twin of
the same widths, MSE on the +-1 labels, constant step, no momentum or weight
decay, 3,000 steps with 10 log-spaced checkpoints; learning rates
{0.01, 0.03, 0.1, 0.2}, seeds 0-4; one GPU per cell, ~1 h at n = 50,000):

```bash
for N in 500 1000 2000 5000 10000 20000 50000; do
for LR in 0.01 0.03 0.1 0.2; do for S in 0 1 2 3 4; do
python scripts/sweep_gd_baseline.py --conf conf/gd_baseline_3k_ffn_3l.yaml \
  --override lr=${LR} dataset.n_train=${N} seed=${S}
done; done; done
```

**What the passes select** (the pass-overlap figure): the first-order arm at k = 512,
n = 10,000, r/k in {2, 5, 10} %, 6 passes, beta = 0.25, seeds 0-2, recording
the overlap of every pass's read/write directions with the earlier ones:

```bash
for S in 0 1 2; do
python scripts/probe_rowcol_pass_overlap.py --k 512 --n 10000 \
  --r-fracs 0.02 0.05 0.10 --passes 6 --seed ${S} --device cuda:0
done
```

## Convolutional network (test error against n, right panel; GD training curves)

Architecture (`conf/backward_rowcol_cnn_*.yaml`): conv 3x3 with 4,096
channels (no filter) -> filter k_c -> conv 3x3 with 4,096 channels, 4x4 max
pooling -> filter k_3, flatten -> fully connected 5,000 -> ridge. One
(k_c, k_3, n, seed) job = forward fit (checkpointed) + every (r, pass) cell.
Seeds 0-4; n in {500, 1000, 2000, 5000, 10000, 20000, 50000}; one GPU per job
(about 1.5 h at n = 50,000 on an H100).

**BackLoFi, first-order signal** (`rcfopci`; k_c in {64, 128, 256, 512, 1024},
k_3 in {32, 64, 128, 256, 512}; r = round(f min(k_c, k_3)) and round(f k_c) at
the two boundaries, f in {5, 10, 15, 20, 25} %, 4 passes, beta = 4):

```bash
for N in 500 1000 2000 5000 10000 20000 50000; do for S in 0 1 2 3 4; do
for KC in 64 128 256 512 1024; do for K3 in 32 64 128 256 512; do
python scripts/sweep_backward_rowcol_cnn.py \
  --conf conf/backward_rowcol_cnn_first_order.yaml \
  --override job=kc${KC}_k3${K3} dataset.n_train=${N} seed=${S} device=cuda \
             ckpt_dir=results/backward_rowcol_cnn_sweep/checkpoints
done; done; done; done
```

**BackLoFi, exact signal** (`rcexpci`; k_c, k_3 in {128, 256, 512, 1024},
f in {10, 20, 25} %, 3 passes): the same loops with
`--conf conf/backward_rowcol_cnn_exact.yaml` and `KC`, `K3` in
`128 256 512 1024`.

**Random-features network** (identity filters, seeds 0-4):

```bash
for N in 500 1000 2000 5000 10000 20000 50000; do for S in 0 1 2 3 4; do
python scripts/sweep_baseline_rf_cnn.py --conf conf/baselines_rf_cnn.yaml \
  --override dataset.n_train=${N} seed=${S}
done; done
```

**Full-batch gradient descent** (trainable twin of the same widths, MSE on
the +-1 labels, constant step, no momentum or weight decay, 5,000 steps with
the checkpoint steps fixed in the config, bfloat16 autocast; the whole training
set is one batch, accumulated over chunks). Learning rates {0.001, 0.003,
0.005} (0.01 diverges at the third step); seeds 0-2 for 0.003 and 0.005 at
every n and for 0.001 at n <= 2,000, seed 0 for 0.001 above. One GPU suffices
up to n = 2,000 (`~2 h` at n = 2,000); larger n were run data-parallel on
4 GPUs with `torchrun` (the gradient is summed over ranks, so the step is the
same full-batch step; ~25 h at n = 50,000 on 4 H100). A finished cell is
extended from its checkpoint when `max_steps` is raised.

```bash
# n <= 2000, one GPU
for N in 500 1000 2000; do for LR in 0.001 0.003 0.005; do for S in 0 1 2; do
python scripts/sweep_gd_baseline_cnn.py --conf conf/gd_baseline_cnn.yaml \
  --override dataset.n_train=${N} lr=${LR} seed=${S}
done; done; done
# n >= 5000, 4 GPUs of one node (seeds 0 1 2 for lr 0.003 and 0.005, seed 0 for 0.001)
torchrun --standalone --nproc_per_node=4 scripts/sweep_gd_baseline_cnn.py \
  --conf conf/gd_baseline_cnn.yaml --override dataset.n_train=${N} lr=${LR} seed=${S}
```

`scripts/run_gd_baseline_cnn_local.py` runs the same cells two at a time on a
two-GPU workstation.

## Figures

The numbers every figure uses are shipped in `data/` (see `data/README.md`);
adding `--from-data` to the commands below redraws the figures from them, and
`--dump-data` regenerates `data/` from `results/` after rerunning the experiments.

```bash
python scripts/plotting/plot_paper_test_error_vs_n.py      # test_error_vs_n_ffn, test_error_vs_n_cnn
python scripts/plotting/plot_paper_passes_ffn.py           # passes_ffn
python scripts/plotting/plot_paper_dense_ffn.py            # dense_ffn
python scripts/plotting/plot_paper_appendix_g.py --gd-cnn-ns 1000 5000 10000 50000
                                                           # pass_overlap, gd_curves_ffn, gd_curves_cnn
```

Each script prints the selection table (best configuration per n with the
seed mean and standard error) and writes `imgs/paper/<name>.{pdf,png}`.

## Running on a cluster

`slurm/array_template.run` is a generic SLURM array template: it maps
`SLURM_ARRAY_TASK_ID` onto a grid and calls one experiment script per task.
The results root can be redirected with `RESULTS_BASE_PATH` (see
`scripts/config_utils.py`), e.g. to a scratch file system.
