# Numbers behind the figures

Seed-level test errors (in percent, binary CIFAR-10 animal vs. vehicle, 10,000
test images) of every cell the paper's figures use, written by the plotters
with `--dump-data` and read back with `--from-data`.

| file | figure | cells |
|---|---|---|
| `test_error_vs_n_ffn.csv` | test error vs n, fully connected | forward LoFi, BackLoFi first-order (`rcfopcit`) and exact (`rcexpcit`), random features (`rf`), ridge on the pixels (`linear`) |
| `test_error_vs_n_cnn.csv` | test error vs n, convolutional | forward LoFi, BackLoFi first-order (`rcfopci`) and exact (`rcexpci`), random features (`rf`) |
| `gd_curves_ffn.csv`, `gd_curves_cnn.csv` | GD in both figures above and the GD training curves | full-batch GD test error at every recorded step, per (n, learning rate, seed) |
| `passes_ffn.csv` | number of passes at small r | 10-pass first-order arm (`rcfopcitm`) and forward LoFi |
| `dense_ffn.csv` | damped update | damped arm (`rcfodia10`), 4-pass swap arm (`rcfopcit`) and forward LoFi |
| `pass_overlap/overlap_seed*.json` | what the passes select | output of `scripts/probe_rowcol_pass_overlap.py`, one file per seed |

Cell files have the columns `n, width, arm, r, pass, idx, test_error`: `width`
is the filter rank (`k` for the fully connected network, `kc x k3` for the
convolutional one, empty for the pixel/random-features baselines), `arm` the
arm tag (`fwd` for forward LoFi), `r` the rank fraction in percent and `pass`
the pass (empty for arms without passes), `idx` the position of the seed in
the cell. GD files have the columns `n, lr, seed, step, test_error`.
