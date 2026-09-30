"""Back-propagation trainer supporting layerwise and end-to-end training."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from neural_lofi.models.backprop import BackpropModel
from neural_lofi.utils.metrics import accuracy

from .base import BaseTrainer, BaseTrainerConfig

log = logging.getLogger(__name__)


@dataclass
class BackpropConfig(BaseTrainerConfig):
    """Configuration for gradient-based trainers.

    Parameters
    ----------
    epochs : int
        Number of training epochs (per layer group in layerwise mode).
    lr : float
        Learning rate passed to the optimizer.
    optimizer_cls : type
        Optimizer class (must accept ``params`` and ``lr`` arguments).
    mode : {"layerwise", "end_to_end"}
        Greedy layerwise training (default) trains one parameter group at a
        time while freezing the rest; ``end_to_end`` trains all parameters
        jointly via a single parameter group.
    weight_decay : float
        Weight decay forwarded to ``optimizer_cls``.
    optimizer_kwargs : dict
        Extra keyword arguments forwarded to ``optimizer_cls`` (e.g. ``momentum``).
    """

    epochs: int = 10
    lr: float = 1e-3
    optimizer_cls: type = torch.optim.Adam
    mode: Literal["layerwise", "end_to_end"] = "layerwise"
    weight_decay: float = 0.0
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)


class BackpropTrainer(BaseTrainer):
    """Gradient-descent trainer supporting layerwise and end-to-end modes.

    With ``config.mode == "layerwise"`` (default) each parameter group is
    trained for ``config.epochs`` epochs while all other parameters are frozen,
    and the final classifier is trained last.  With ``config.mode ==
    "end_to_end"`` a single group containing every trainable parameter is
    optimized jointly.

    Parameters
    ----------
    grad_clip_norm : float | None
        If set (and > 0), clip the gradient norm of the active group's
        parameters to this value before each optimizer step.
    warmup_fraction : float | None
        If set, apply a linear-warmup + cosine-annealing learning rate schedule
        over each group's total steps (``epochs * len(loader)``); the warmup
        covers ``warmup_fraction`` of those steps and the rest is cosine decay.
        When ``None`` the optimizer learning rate is held constant.
    """

    config: BackpropConfig

    def __init__(
        self,
        model: nn.Module,
        config: BackpropConfig,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        pred_fn: Callable[[Tensor], Tensor] | None = None,
        layer_groups: list[list[nn.Parameter]] | None = None,
        grad_clip_norm: float | None = None,
        warmup_fraction: float | None = None,
    ) -> None:
        super().__init__(model, config)
        self.loss_fn = loss_fn
        self.pred_fn: Callable[[Tensor], Tensor] = (
            pred_fn if pred_fn is not None else lambda p: p.argmax(dim=-1)
        )
        self.layer_groups = (
            layer_groups if layer_groups is not None else self._default_groups()
        )
        self.grad_clip_norm = grad_clip_norm
        self.warmup_fraction = warmup_fraction

    def _scheduled_lr(self, step: int, total_steps: int, warmup_steps: int) -> float:
        """Linear warmup then cosine annealing, anchored at ``config.lr``."""
        base_lr = self.config.lr
        if step < warmup_steps:
            return base_lr * (step / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Train each parameter group in turn (one group only in end-to-end mode).

        Optional ``kwargs``: ``test_loader`` for per-epoch test accuracy,
        ``epoch_callbacks`` (called as ``cb(self, epoch, entry)`` after each
        epoch), and ``step_callbacks`` (called as ``cb(self, step_info)`` after
        every optimizer step, where ``step_info`` carries ``global_step``,
        ``layer``, ``epoch``, ``batch_in_epoch``, ``loss``, ``lr``, and
        ``accuracy``).
        """
        test_loader: DataLoader | None = kwargs.get("test_loader")  # type: ignore[assignment]
        epoch_callbacks: list[Callable[..., Any]] | None = kwargs.get("epoch_callbacks")  # type: ignore[assignment]
        step_callbacks: list[Callable[..., Any]] | None = kwargs.get("step_callbacks")  # type: ignore[assignment]
        epochs = self.config.epochs

        history: list[dict[str, Any]] = []
        global_step = 0

        for layer_idx, params in enumerate(self.layer_groups):
            if not params:
                continue

            self._freeze_all()
            for p in params:
                p.requires_grad = True

            optimizer = self.config.optimizer_cls(
                (p for p in params if p.requires_grad),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_kwargs,
            )

            # Per-group LR schedule: warmup + cosine over this group's steps.
            total_steps = max(epochs * max(len(loader), 1), 1)
            warmup_steps = (
                max(1, int(total_steps * self.warmup_fraction))
                if self.warmup_fraction is not None
                else 0
            )
            step_count = 0

            self.model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                total_grad_norm = 0.0
                all_y_true: list[np.ndarray] = []
                all_y_pred: list[np.ndarray] = []

                for batch_idx, (x, y) in enumerate(loader, start=1):
                    x, y = self._to_device_safe(x, y)
                    optimizer.zero_grad(set_to_none=True)
                    pred = self.model(x)
                    loss = self.loss_fn(pred.squeeze(), y)
                    loss.backward()

                    if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)

                    total_grad_norm += self._global_grad_norm()
                    optimizer.step()
                    batch_loss = loss.item()
                    total_loss += batch_loss

                    if self.warmup_fraction is not None:
                        lr_now = self._scheduled_lr(
                            step_count, total_steps, warmup_steps
                        )
                        for group in optimizer.param_groups:
                            group["lr"] = lr_now
                    step_count += 1
                    global_step += 1

                    y_true_np = y.detach().cpu().numpy()
                    y_pred_np = self.pred_fn(pred.detach()).cpu().numpy()
                    all_y_true.append(y_true_np)
                    all_y_pred.append(y_pred_np)

                    if step_callbacks:
                        step_info: dict[str, Any] = {
                            "global_step": global_step,
                            "layer": layer_idx,
                            "epoch": epoch,
                            "batch_in_epoch": batch_idx,
                            "loss": batch_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                            "accuracy": accuracy(y_true_np, y_pred_np),
                        }
                        for cb in step_callbacks:
                            cb(self, step_info)

                n_batches = max(len(loader), 1)
                mean_loss = total_loss / n_batches
                lr = optimizer.param_groups[0]["lr"]
                train_acc = accuracy(
                    np.concatenate(all_y_true), np.concatenate(all_y_pred)
                )

                entry: dict[str, Any] = {
                    "layer": layer_idx,
                    "epoch": epoch,
                    "loss": mean_loss,
                    "accuracy": train_acc,
                    "grad_norm": total_grad_norm / n_batches,
                    "lr": lr,
                }

                test_metrics = self._eval_on_test(test_loader)
                if test_metrics:
                    entry.update({f"test_{k}": v for k, v in test_metrics.items()})

                if epoch_callbacks:
                    for cb in epoch_callbacks:
                        cb(self, epoch, entry)

                history.append(entry)

                if self.config.verbose:
                    msg = (
                        f"Layer {layer_idx + 1}/{len(self.layer_groups)} "
                        f"Epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}"
                    )
                    if test_metrics:
                        parts = [f"{k}={v:.4f}" for k, v in test_metrics.items()]
                        msg += f" test: [{', '.join(parts)}]"
                    log.info(msg)

        return self.model, {"epochs": history}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_groups(self) -> list[list[nn.Parameter]]:
        """Pick the parameter grouping for the wrapped model and training mode.

        ``end_to_end`` mode trains every parameter jointly (a single group);
        ``layerwise`` mode emits one group per weighted (``conv`` /
        ``fully_connected``) layer plus a final classifier group.
        """
        if self.config.mode == "end_to_end":
            return [list(self.model.parameters())]
        if isinstance(self.model, BackpropModel):
            return self._groups_from_model(self.model)
        raise TypeError(
            "BackpropTrainer layerwise mode requires a BackpropModel. "
            "Pass layer_groups for custom models, or use mode='end_to_end'."
        )

    @staticmethod
    def _groups_from_model(model: BackpropModel) -> list[list[nn.Parameter]]:
        """One group per trainable (conv/linear) layer plus a classifier group.

        Type-driven (no spec lookup): a trainable layer is an ``nn.ModuleDict``
        (``conv``/``linear`` + ``activation``); stateless ops (pool / flatten /
        l2norm) carry no parameters and are skipped.
        """
        groups: list[list[nn.Parameter]] = []
        for layer in model.layers:
            if isinstance(layer, nn.ModuleDict):
                op = layer["conv"] if "conv" in layer else layer["linear"]
                groups.append(list(op.parameters()))
        groups.append(list(model.classifier.parameters()))
        return groups

    def _freeze_all(self) -> None:
        """Set ``requires_grad=False`` on every model parameter."""
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _eval_on_test(self, test_loader: DataLoader | None) -> dict[str, float] | None:
        """Return ``{"accuracy": ...}`` on ``test_loader``, or ``None`` if not given."""
        if test_loader is None:
            return None
        self.model.eval()
        all_y: list[np.ndarray] = []
        all_p: list[np.ndarray] = []
        for x, y in test_loader:
            x, y = self._to_device_safe(x, y)
            pred = self.model(x)
            all_y.append(y.cpu().numpy())
            all_p.append(self.pred_fn(pred).cpu().numpy())
        self.model.train()
        return {"accuracy": accuracy(np.concatenate(all_y), np.concatenate(all_p))}

    def _global_grad_norm(self) -> float:
        """L2 norm of gradients over all parameters with ``.grad`` set."""
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(p.grad.data.norm(2).item()) ** 2
        return float(total**0.5)

    def _to_device_safe(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        """Move tensors to ``config.device``; downcast float64 -> float32 for GPU."""
        device = self.config.device
        out: list[Tensor] = []
        for t in tensors:
            if t.is_floating_point() and t.dtype == torch.float64:
                out.append(t.to(device=device, dtype=torch.float32))
            else:
                out.append(t.to(device=device))
        return tuple(out)
