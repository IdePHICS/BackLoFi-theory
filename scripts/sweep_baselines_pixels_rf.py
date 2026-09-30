"""Two baselines for one ``(n_train, seed)``: linear ridge on the pixels and a
3-layer random-features network (identity reduces, no LoFi filter).

Files in ``output_dir``::

    {preset}_seed{s}_ntrain{n}_ntest{t}_D{D}_linear.json    LOO-ridge on flattened x
    {preset}_seed{s}_ntrain{n}_ntest{t}_L{L}_{arch}k0{act}_rf.json   random features

Both use the training-only leave-one-out ridge of ``fit_linear_readout`` and
report the test error on the fixed test split.
"""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _block_model import build_block_model
from _run_utils import save_results
from config_utils import parse_classes, with_config
from torch import nn
from torch.utils.data import DataLoader

from neural_lofi.datasets import load_tensors
from neural_lofi.training import SpectralTrainer, SpectralTrainerConfig
from neural_lofi.training.base import fit_linear_readout
from neural_lofi.utils.helpers import resolve_device, set_seed

log = logging.getLogger(__name__)


@with_config("conf/baselines_pixels_rf.yaml", use_timestamp=False)
def main(cfg, out_dir, *, force_run: bool = False) -> None:
    d = cfg.dataset
    base = f"{d.preset}_seed{cfg.seed}_ntrain{d.n_train}_ntest{d.n_test}"
    arch = cfg.get("arch_tag") or f"p{cfg.all_p}"
    rf_path = out_dir / f"{base}_L{len(cfg.blocks)}_{arch}k0{cfg.activation}_rf.json"

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
        flatten=True,
    )
    X_train, y_train = load_tensors(
        split=str(d.get("train_split", "train")), n_samples=d.n_train, **common
    )
    X_test, y_test = load_tensors(split="test", n_samples=d.n_test, **common)
    lin_path = out_dir / f"{base}_D{X_train.shape[1]}_linear.json"
    if lin_path.exists() and rf_path.exists() and not force_run:
        log.info(
            "[n=%d seed=%d] both baselines exist — nothing to do.", d.n_train, cfg.seed
        )
        return
    ridge = dict(
        alpha_min=cfg.alpha_min, alpha_max=cfg.alpha_max, alpha_num=cfg.alpha_num
    )

    # 1. linear ridge on the flattened, normalised pixels
    if force_run or not lin_path.exists():
        metrics = fit_linear_readout(
            nn.Module(),
            X_train.to(device),
            y_train.to(device).float().reshape(-1),
            h_test=X_test.to(device),
            y_test_np=y_test.float().reshape(-1).cpu().numpy(),
            verbose=False,
            **ridge,
        )
        save_results(lin_path, {"final": metrics, "baseline": "linear_pixels"})
        log.info(
            "[n=%d seed=%d] linear: test_acc=%s",
            d.n_train,
            cfg.seed,
            metrics.get("test_accuracy"),
        )

    # 2. 3-layer random features, identity reduces
    if force_run or not rf_path.exists():
        train_loader = DataLoader(
            list(zip(X_train, y_train)), batch_size=cfg.batch_size
        )
        test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=cfg.batch_size)
        set_seed(cfg.seed)
        model = build_block_model(
            cfg, input_shape=tuple(X_train.shape[1:]), seed=cfg.seed
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
