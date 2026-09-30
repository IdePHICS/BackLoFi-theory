"""Random-features baseline of the CNN row-then-column campaign for one
``(n_train, seed)``: the same 2-conv + avg-pool + FC architecture with
identity reduces everywhere (no LoFi filter), ridge on the FC outputs.

File in ``output_dir``::

    {preset}_seed{s}_ntrain{n}_ntest{t}_L{L}_{arch}k0{act}_rf.json

(the linear-ridge-on-pixels baseline is the FFN campaign's, same task and
splits: results/backward_rowcol_sweep/cifar10_ffn/baselines).
"""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _block_model import build_block_model
from _run_utils import save_results
from config_utils import parse_classes, with_config
from torch.utils.data import DataLoader

from neural_lofi.datasets import load_tensors
from neural_lofi.training import SpectralTrainer, SpectralTrainerConfig
from neural_lofi.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)


@with_config("conf/baselines_rf_cnn.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    d = cfg.dataset
    arch = cfg.get("arch_tag") or f"c{cfg.conv_p}p{cfg.fc_p}"
    rf_path = (
        out_dir / f"{d.preset}_seed{cfg.seed}_ntrain{d.n_train}_ntest{d.n_test}"
        f"_L{len(cfg.blocks)}_{arch}k0{cfg.activation}_rf.json"
    )
    if rf_path.exists() and not force_run:
        log.info("[n=%d seed=%d] baseline exists — nothing to do.", d.n_train, cfg.seed)
        return
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    common: dict[str, Any] = dict(
        dataset_type=d.name,
        device=device,
        seed=cfg.seed,
        root=cfg.root,
        class_preset=d.get("preset"),
        classes=parse_classes(cfg),
        remap_labels=cfg.remap_labels,
        flatten=False,
    )
    X_train, y_train = load_tensors(
        split=str(d.get("train_split", "train")), n_samples=d.n_train, **common
    )
    X_test, y_test = load_tensors(split="test", n_samples=d.n_test, **common)
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=cfg.batch_size)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=cfg.batch_size)
    set_seed(cfg.seed)
    model = build_block_model(cfg, input_shape=tuple(X_train.shape[1:]), seed=cfg.seed)
    ridge = dict(
        alpha_min=cfg.alpha_min, alpha_max=cfg.alpha_max, alpha_num=cfg.alpha_num
    )
    _model, results = SpectralTrainer(
        model, SpectralTrainerConfig(device=device, verbose=False, **ridge)
    ).fit(train_loader, test_loader=test_loader)
    results["baseline"] = "random_features"
    save_results(rf_path, results)
    log.info(
        "[n=%d seed=%d] random features: test_acc=%s",
        d.n_train,
        cfg.seed,
        results["final"].get("test_accuracy"),
    )


if __name__ == "__main__":
    main()
