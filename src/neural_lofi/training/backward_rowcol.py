"""Row-then-column backward correction (the backward step of the paper).

Implements the row-then-column update on a fitted forward Neural LoFi model,
as a standalone trainer that touches no existing code.  One round is a top-down sweep over the boundaries
``ℓ | ℓ+1`` (blocks ``j | j+1``); at each boundary:

1. *Directions.*  ``A_ℓ = E[y g^{ℓ+1} (z^{ℓ-1})ᵀ]`` (line 3), thin SVD, the
   top ``r`` pairs: write-directions ``U`` in layer ℓ+1, read-directions
   ``V_←`` in layer ℓ−1.  ``g^{ℓ+1}`` is the *local gate*
   ``σ'(h^{ℓ+1})/√p_{ℓ+1}`` (Algorithm 1 line 8, unit gain), at every
   boundary including the top, so the readout never enters a round.
2. *Frozen scores.*  ``s = Uᵀ g^{ℓ+1}`` and ``ỹ_i = y s_i`` (line 4,
   Approx. "Frozen upper pass").
3. *Rows of layer ℓ* (lines 5–6).  Drop the ``r`` weakest columns of the
   layer's reduce and append ``V_←`` with fresh ``randn`` lift rows; the
   width never changes.  Re-evaluate the layer: ``z̃^ℓ``.
4. *Columns of layer ℓ+1* (line 7).  ``Q̃ = E[z̃^ℓ (y s)ᵀ]``; drop the ``r``
   weakest columns of the upper reduce and append ``Q̃`` with lift rows
   ``η u_i``.

*Weakness* is one criterion for every column whatever its origin: the
signed-covariance Rayleigh quotient ``|vᵀ E[y z zᵀ] v| / ‖v‖²`` on the
layer's current input — for a forward eigenvector on unchanged input this
is its eigenvalue, so pass 1 on a fresh model drops the weakest forward
columns, and later passes drop whatever is weakest, forward or backward.

*Scale* is the one thing the derivation leaves open (its Assessment).  A
single dimensionless ``β`` fixes both appends by the convention of the
earlier campaigns, "one backward coordinate = β forward coordinates": each
appended read-direction is rescaled so its coordinate RMS equals ``β`` times
the mean forward-coordinate RMS of the layer, and the appended block gets
one common factor ``η`` so that the mean over channels of its per-neuron
RMS equals the same target, keeping the gradient's relative amplitudes
across channels and the bias.

Passes re-run the round on the corrected network (the readout is refitted
after every round).  Scope: flat FFN blocks, dense lifts, no whitening,
scalar labels.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.spectral import SpectralModel, _Block, _Reduce
from ..utils.transforms import DenseTransform
from .backward_lofi import block_derivative_features
from .eigen import (
    covariance_chunk_signed_accumulate_,
    linear_mean_prepend_block_from_covariance,
)
from .spectral import SpectralTrainer, SpectralTrainerConfig

log = logging.getLogger(__name__)

EPS = 1e-12
_CHAINS = ("local", "mean_field", "exact", "first_order")
_COLUMN_SCALES = ("mean", "max", "per_channel")
_SWAP_RULES = ("weakest", "replace_own", "importance")
_TAU_OVER = ("all", "random_lift")
_UPPER_UPDATES = ("column", "block")
FWD, READ, PAIR = "fwd", "read", "pair"


@dataclass
class RowColConfig(SpectralTrainerConfig):
    """Configuration for :class:`RowColBackwardTrainer`.

    Parameters
    ----------
    r_frac
        Rank of every boundary as a fraction of the configured forward
        cutoff ``k`` (``r = round(r_frac·k)``); the number of columns replaced
        by each append.
    beta
        Scale of the appended coordinates in units of one forward coordinate
        (both appends).
    n_passes
        Rounds; each starts from the network the previous one produced.
    chain
        Backward signal: ``"local"`` (the block's own gate, unit gain, at
        every boundary; the readout never enters a round) or ``"mean_field"``
        (``g^L = a ⊙ σ'(h^L)/√p`` from the readout, then Algorithm 2 line 8
        through the corrected block with the random branch replaced by its
        per-neuron sample mean and the paired branch kept per sample —
        Approx. "Mean-field backward signal") or ``"exact"`` (the same
        top-down chain with the random branch kept per sample: the
        paragraph "Exact chain (top-down)", eq. r-corrected, no
        approximation beyond the frozen upper signal) or ``"first_order"``
        (Approx. 6' "First-order backward signal": the mean-field gain
        through the PHASE-I block above, all its columns, plus the spike
        term of the boundary above evaluated at order zero — the frozen
        scores along the write-directions times the read-directions lifted
        through the Phase-I block ``j``, ``q⁰_i = W^{j} v_i``, each scaled
        by the per-channel convention of the installed columns; the
        installed paired columns and the row step's appended directions
        are not seen.  "Phase I" is the network at the start of the pass.
        For L = 3 the local signal of the top block is the exact one, so
        the triplets and scores are those of the boundary above.)
    chunk_size
        Sample chunk for every streamed statistic.
    svd_oversample, svd_niter
        Randomized thin SVD of ``A_ℓ`` (``torch.svd_lowrank``).
    lift_seed
        Seed of the fresh lift rows and of the randomized SVD.
    column_scale
        How the common factor ``η`` of the appended block is set from the
        per-channel coordinate RMS: ``"mean"`` (mean channel = β forward
        coordinates, the gradient's relative amplitudes kept), ``"max"``
        (the strongest channel = β forward coordinates, relative amplitudes
        kept, nothing louder than a forward coordinate) or ``"per_channel"``
        (every channel = β forward coordinates, amplitudes equalised as in
        the earlier column-coupled convention).
    swap_rule
        Which columns a swap removes.  ``"weakest"``: the ``r`` lowest
        Rayleigh quotients of the layer, whatever their origin — backward
        columns score high, so passes accumulate them and consume the
        forward features.  ``"replace_own"``: the row step replaces the
        layer's read segment and the column step the upper layer's paired
        segment (at pass 1, the weakest *forward* columns), so every pass
        re-estimates the same slots: the fixed-point iteration.
        ``"importance"``: the ``r`` columns of lowest gradient importance
        (section "Feature importance at fixed budget"): the RMS over neurons
        of the rate ``e_ℓ λ_j(v) c_j + d_ℓ ℓ_j(v)`` at the current lift
        coefficients, with ``λ_j, ℓ_j`` the projections of
        ``Λ₂(v) = E[y ĝ ⟨v,z⟩²]`` and ``Λ₁(v) = E[y ĝ ⟨v,z⟩]`` on the neuron's
        output column and ``ĝ`` the retained backward feature; uniform over
        forward, read and paired columns, every pass.
    tau_over
        Which surviving columns define the forward-coordinate scale ``τ`` of
        a swap.  ``"all"``: every survivor (the first campaigns; paired
        columns carry √p-larger coordinates by construction, so ``τ`` inflates
        from pass 2 on).  ``"random_lift"``: the surviving forward and read
        columns only — the columns lifted by random rows, i.e. the "forward
        channels" the rule "one backward coordinate = β forward coordinates"
        refers to; ``τ`` then stays at the forward scale at every pass.
    upper_update
        What a boundary does to block ``ℓ+1``.  ``"column"``: the column step
        only (Algorithm 2 line 8).  ``"block"``: Algorithm 3 ("Closing the
        step: the rows of layer ℓ+1") — before the swap, every feature of the
        block is re-evaluated on the updated input ``z̃^ℓ``, lifts unchanged:
        the forward group is refitted as forward LoFi of ``z̃^ℓ`` and brought
        to the old gauge by the ``Σ_z``-weighted orthogonal Procrustes
        rotation ``E Fᵀ`` of ``V_cᵀ Σ_z V_old``; every surviving read and
        paired column, of this pass or an earlier one, is recomputed as
        ``E[y s_i z̃^ℓ]`` with the stored scores of the boundary that created
        it and rescaled to its per-channel target.  With the fixed-width swap
        the forward group has fewer survivors than the refit returns: the
        swapped-out forward columns are kept as *ghosts* — alignment targets
        that make the Procrustes problem square, never installed — and only
        the survivors are written.  Gates, signals, scores and the upper
        activations keep their one-step lag.  Requires
        ``column_scale="per_channel"``.
    """

    r_frac: float = 0.25
    beta: float = 1.0
    n_passes: int = 1
    chain: str = "local"
    chunk_size: int = 8192
    svd_oversample: int = 32
    svd_niter: int = 4
    lift_seed: int = 0
    column_scale: str = "mean"
    swap_rule: str = "weakest"
    tau_over: str = "all"
    upper_update: str = "column"


def _sigma_prime(h: Tensor, activation: str) -> Tensor:
    """Derivative of the block activation, elementwise."""
    if activation == "relu":
        return (h > 0).to(h.dtype)
    if activation == "tanh":
        return 1.0 - torch.tanh(h).pow(2)
    if activation == "sigmoid":
        sg = torch.sigmoid(h)
        return sg * (1.0 - sg)
    raise NotImplementedError(f"σ' for activation {activation!r}")


def _rms_cols(z: Tensor, v: Tensor, chunk: int) -> Tensor:
    """Per-column RMS of the coordinates ``z @ v`` over samples (float64)."""
    acc = torch.zeros(v.shape[1], dtype=torch.float64, device=z.device)
    for zc in z.split(chunk):
        acc += (zc @ v).double().pow(2).sum(dim=0)
    return (acc / z.shape[0]).sqrt()


def _rayleigh(z: Tensor, y: Tensor, v: Tensor, chunk: int) -> Tensor:
    """``|E[y ⟨v, z⟩²]| / ‖v‖²`` per column: the signed-covariance quotient."""
    acc = torch.zeros(v.shape[1], dtype=torch.float64, device=z.device)
    for zc, yc in zip(z.split(chunk), y.split(chunk), strict=True):
        c = (zc @ v).double()
        acc += (yc.double().unsqueeze(1) * c * c).sum(dim=0)
    return (acc / z.shape[0]).abs() / v.double().norm(dim=0).pow(2).clamp_min(EPS)


def _cross(a: Tensor, b: Tensor, y: Tensor, chunk: int) -> Tensor:
    """``E[y a bᵀ]`` for row-aligned samples ``a`` (n, p) and ``b`` (n, q)."""
    out = torch.zeros(a.shape[1], b.shape[1], dtype=torch.float64, device=a.device)
    for ac, bc, yc in zip(a.split(chunk), b.split(chunk), y.split(chunk), strict=True):
        out += ac.double().transpose(0, 1) @ (yc.double().unsqueeze(1) * bc.double())
    return out / a.shape[0]


def _reduce_cols(red: _Reduce) -> Tensor:
    """Reduce columns as ``(p, K)`` (a conv reduce stores a 1×1 kernel)."""
    v = red.V
    return v.reshape(v.shape[0], v.shape[1]).transpose(0, 1) if v.ndim == 4 else v


def _second_moments(z: Tensor, y: Tensor, chunk: int) -> tuple[Tensor, Tensor, Tensor]:
    """``(E[y z zᵀ], E[y z], E[z zᵀ])`` of the row samples ``z`` (float64)."""
    p = z.shape[1]
    c_y = torch.zeros(p, p, dtype=torch.float64, device=z.device)
    sigma = torch.zeros_like(c_y)
    m_y = torch.zeros(p, dtype=torch.float64, device=z.device)
    m_1 = torch.zeros_like(m_y)
    t_y = t_1 = 0
    for zc, yc in zip(z.split(chunk), y.split(chunk), strict=True):
        t_y = covariance_chunk_signed_accumulate_(c_y, m_y, zc, yc, t_y)
        t_1 = covariance_chunk_signed_accumulate_(
            sigma, m_1, zc, torch.ones_like(yc), t_1
        )
    return (
        0.5 * (c_y + c_y.transpose(0, 1)),
        m_y,
        0.5 * (sigma + sigma.transpose(0, 1)),
    )


def refit_aligned(
    c_y: Tensor, m_y: Tensor, sigma: Tensor, old: Tensor, rcfg: Any
) -> tuple[Tensor, dict[str, float]]:
    """Forward-LoFi filter of the current input in the gauge of ``old``.

    ``V_c = FwdLoFi`` from the signed moments ``(c_y, m_y)`` with the reduce's
    own rule (first-moment column, deflation); ``old`` ``(p, K)`` is the old
    forward group ``[survivors | ghosts]``.  Returns ``V_c E Fᵀ`` with
    ``V_cᵀ Σ_z old = E S Fᵀ`` (eq. procrustes: the basis of the refitted
    subspace that changes the realised preactivations least) and the
    diagnostics of eq. principal-angles on the unweighted cosines.
    """
    _vals, v_c = linear_mean_prepend_block_from_covariance(
        c_y,
        m_y,
        k=int(rcfg.k),
        include_linear_mean=bool(rcfg.include_linear_mean),
        orthogonalize=bool(rcfg.orthogonalize_to_mean),
        target_dtype=torch.float64,
        target_device=c_y.device,
        log_prefix="block refit: ",
    )
    old = old.double()
    if v_c.shape != old.shape:
        raise RuntimeError(
            f"refit returned {tuple(v_c.shape)}, forward group is {tuple(old.shape)}"
        )
    e_mat, _s, f_h = torch.linalg.svd(v_c.transpose(0, 1) @ sigma @ old)
    new = v_c @ (e_mat @ f_h)
    cos = torch.linalg.svdvals(v_c.transpose(0, 1) @ old).clamp(max=1.0)
    return new, {
        "angle_sum": float((1.0 - cos).sum()),
        "cos_min": float(cos.min()),
    }


class RowColBackwardTrainer(SpectralTrainer):
    """Row-then-column correction of a fitted forward model (Algorithm 2).

    ``fit(loader, test_loader=..., pass_callback=...)`` deep-copies the model,
    runs ``n_passes`` rounds and returns ``(corrected, results)`` for the
    last pass; ``pass_callback(p, results)`` is called after every round.
    ``results["backward"]["origins"]`` records, per block, the origin of
    every reduce column (``fwd`` / ``read`` / ``pair``).
    """

    def __init__(self, model: SpectralModel, config: RowColConfig) -> None:
        if not 0.0 < config.r_frac < 1.0:
            raise ValueError(f"r_frac must be in (0, 1), got {config.r_frac}")
        if config.beta <= 0.0:
            raise ValueError(f"beta must be positive, got {config.beta}")
        if config.n_passes < 1:
            raise ValueError(f"n_passes must be >= 1, got {config.n_passes}")
        if config.chain not in _CHAINS:
            raise ValueError(f"chain must be one of {_CHAINS}, got {config.chain!r}")
        if config.swap_rule not in _SWAP_RULES:
            raise ValueError(
                f"swap_rule must be one of {_SWAP_RULES}, got {config.swap_rule!r}"
            )
        if config.column_scale not in _COLUMN_SCALES:
            raise ValueError(
                f"column_scale must be one of {_COLUMN_SCALES}, "
                f"got {config.column_scale!r}"
            )
        if config.tau_over not in _TAU_OVER:
            raise ValueError(
                f"tau_over must be one of {_TAU_OVER}, got {config.tau_over!r}"
            )
        if config.upper_update not in _UPPER_UPDATES:
            raise ValueError(
                f"upper_update must be one of {_UPPER_UPDATES}, "
                f"got {config.upper_update!r}"
            )
        if config.upper_update == "block" and config.column_scale != "per_channel":
            raise ValueError('upper_update="block" requires column_scale="per_channel"')
        super().__init__(model, config)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def fit(self, loader: DataLoader, **kwargs: object) -> tuple[Any, dict[str, Any]]:
        test_loader = cast("DataLoader | None", kwargs.get("test_loader"))
        pass_callback = cast("Any", kwargs.get("pass_callback"))
        config: RowColConfig = self.config  # type: ignore[assignment]
        dev = torch.device(config.device)

        is_vector, _n_out, _oh = self._peek_label_shape(loader)
        if is_vector:
            raise NotImplementedError("RowColBackwardTrainer supports scalar labels.")
        source: SpectralModel = self.model  # type: ignore[assignment]
        for i, block in enumerate(source.blocks):
            b = cast(_Block, block)
            red = cast(_Reduce, b.reduce)
            if red.is_conv or b.pool is not None:
                raise NotImplementedError(f"Block {i} is convolutional; FFN only.")
            if not isinstance(b.transform, DenseTransform):
                raise NotImplementedError(f"Block {i} needs a dense lift.")
            if red.is_identity:
                raise NotImplementedError(f"Block {i} has an identity reduce.")
            if red.whiten:
                raise NotImplementedError(f"Block {i}: whitened reduces unsupported.")

        model = copy.deepcopy(source).to(dev)
        self.model = model
        n_blocks = len(model.blocks)
        cfgs = model.block_config.blocks
        # Rank per boundary from the configured cutoff (the reduce may carry one
        # more column, the first-moment one).
        self._r = []
        for j in range(n_blocks - 1):
            k_cfg = int(cast(int, cfgs[j].reduce.k))
            width_j = int(
                cast(_Reduce, cast(_Block, model.blocks[j]).reduce).V.shape[1]
            )
            width_up = int(
                cast(_Reduce, cast(_Block, model.blocks[j + 1]).reduce).V.shape[1]
            )
            r = max(1, int(round(config.r_frac * k_cfg)))
            self._r.append(min(r, width_j - 1, width_up - 1))
        self._origin = [
            [FWD] * int(cast(_Reduce, cast(_Block, b).reduce).V.shape[1])
            for b in model.blocks
        ]
        self._gen = torch.Generator(device="cpu").manual_seed(int(config.lift_seed))
        self._init_block_state(model)

        X, y = self._materialise(loader, dev)
        results: dict[str, Any] = {}
        for p in range(1, config.n_passes + 1):
            self._pass = p
            entries = self._round(model, X, y)
            final = self._fit_blocked_readout(loader, test_loader)
            results = {
                "layers": entries,
                "final": final,
                "backward": {
                    "algorithm": "rowcol",
                    "pass": p,
                    "n_passes": config.n_passes,
                    "r_frac": config.r_frac,
                    "beta": config.beta,
                    "chain": config.chain,
                    "column_scale": config.column_scale,
                    "swap_rule": config.swap_rule,
                    "tau_over": config.tau_over,
                    "upper_update": config.upper_update,
                    "ranks": list(self._r),
                    "origins": [
                        {o: tags.count(o) for o in (FWD, READ, PAIR)}
                        for tags in self._origin
                    ],
                },
            }
            if pass_callback is not None:
                pass_callback(p, copy.deepcopy(results))
        return model, results

    # ------------------------------------------------------------------ #
    # One round
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _round(
        self, model: SpectralModel, X: Tensor, y: Tensor
    ) -> list[dict[str, Any]]:
        config: RowColConfig = self.config  # type: ignore[assignment]
        chunk = int(config.chunk_size)
        beta = float(config.beta)
        n_blocks = len(model.blocks)
        feats, gates = self._state(model, X)
        entries: list[dict[str, Any]] = []
        self._ctx: dict[int, tuple[Tensor, Tensor]] = {}
        mean_field = config.chain in ("mean_field", "exact", "first_order")
        # Phase-I network of this pass: the blocks as they are before any
        # correction of the round (the first-order chain reads through them).
        phase1 = copy.deepcopy(model) if config.chain == "first_order" else None
        # Top of the chain: the exact boundary condition g^L = a ⊙ σ'(h^L)/√p.
        g_chain = (
            gates[-1] * self._readout_grad(model).unsqueeze(0) if mean_field else None
        )
        for j in range(n_blocks - 2, -1, -1):
            z_prev = X if j == 0 else feats[j - 1]
            # Signal of block j+1, frozen for this boundary: the local gate, or the
            # chain carried down from the boundary above.
            g_up = g_chain if mean_field else gates[j + 1]
            assert g_up is not None
            r = self._r[j]

            # 1. directions: A = E[y g_up z_prevᵀ], thin SVD
            a_mat = _cross(g_up, z_prev, y, chunk)  # (p_{j+1}, p_{j-1})
            u, s_vals, v_read = self._thin_svd(a_mat, r)

            # 2. frozen scores and modified labels
            scores = (g_up.double() @ u).to(X.dtype)  # (n, r)
            ys = y.unsqueeze(1) * scores
            self._ctx[j] = (g_up, u.to(X.dtype))

            # 3. rows of layer j
            blk = cast(_Block, model.blocks[j])
            red = cast(_Reduce, blk.reduce)
            drop, tau = self._slots(model, j, red.V, z_prev, y, r, chunk, READ)
            v_new = v_read.to(red.V.dtype)
            v_new = v_new * (
                beta * tau / _rms_cols(z_prev, v_new, chunk).clamp_min(EPS)
            ).to(v_new.dtype)
            p_j = int(cast(Tensor, blk.weight).shape[1])
            self._capture_ghosts(j, red.V, drop)
            red.V[:, drop] = v_new
            cast(Tensor, blk.weight)[drop] = torch.randn(
                r, p_j, generator=self._gen
            ).to(cast(Tensor, blk.weight))
            dropped_row = self._retag(j, drop, READ)
            self._register(j, drop, scores, boundary=j)
            feats[j], gates[j] = self._block_state(blk, z_prev, chunk)
            z_tilde = feats[j]

            # 3'. Algorithm 3: every feature of block j+1 on the updated input
            refit = (
                self._update_block(model, j + 1, z_tilde, y, chunk)
                if self._block_mode
                else None
            )

            # 4. columns of layer j+1
            blk_up = cast(_Block, model.blocks[j + 1])
            red_up = cast(_Reduce, blk_up.reduce)
            q_mat = _cross(z_tilde, ys, y.new_ones(y.shape), chunk)  # E[z̃ (y s)ᵀ]
            drop_up, tau_up = self._slots(
                model, j + 1, red_up.V, z_tilde, y, r, chunk, PAIR
            )
            rms_q = _rms_cols(z_tilde, q_mat.to(z_tilde.dtype), chunk)
            p_up = int(cast(Tensor, blk_up.weight).shape[1])
            target = beta * math.sqrt(p_up) * float(tau_up)
            if config.column_scale == "per_channel":
                # every channel = β forward coordinates; amplitudes equalised
                q_mat = q_mat * (target / rms_q.clamp_min(EPS)).unsqueeze(0)
                eta = 1.0
            elif config.column_scale == "max":
                eta = target / float(rms_q.max().clamp_min(EPS))
            else:
                eta = target / float(rms_q.mean().clamp_min(EPS))
            self._capture_ghosts(j + 1, red_up.V, drop_up)
            red_up.V[:, drop_up] = q_mat.to(red_up.V.dtype)
            cast(Tensor, blk_up.weight)[drop_up] = (
                (eta * u).transpose(0, 1).to(cast(Tensor, blk_up.weight))
            )
            dropped_col = self._retag(j + 1, drop_up, PAIR)
            self._register(j + 1, drop_up, scores, boundary=j)
            if phase1 is not None and j > 0:
                g_chain = self._first_order_step(
                    phase1, j, g_up, gates[j], v_read, scores, feats[j], tau_up
                )
            elif mean_field and j > 0:
                g_chain = self._chain_step(
                    model, j, g_up, gates[j], exact=config.chain == "exact"
                )

            entries.append(
                {
                    "boundary": j,
                    "rank": r,
                    "singular_values": [float(x) for x in s_vals[:8].tolist()],
                    "tau_row": float(tau),
                    "tau_col": float(tau_up),
                    "eta": eta,
                    "column_scale": config.column_scale,
                    "swap_rule": config.swap_rule,
                    "rms_q_mean": float(rms_q.mean()),
                    "rms_q_max": float(rms_q.max()),
                    "dropped_row": dropped_row,
                    "dropped_col": dropped_col,
                    **({"refit": refit} if refit is not None else {}),
                }
            )
        return entries

    # ------------------------------------------------------------------ #
    # Pieces
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _readout_grad(self, model: SpectralModel) -> Tensor:
        """``∂f/∂z^L`` of the current readout (constant over samples)."""
        w = model.readout_weight.detach()
        fr = cast(_Reduce, model.final_reduction)
        if fr.is_identity:
            return w.reshape(-1)
        wk = (w / fr.scale if fr.whiten else w).reshape(-1)
        return fr.V @ wk

    @torch.no_grad()
    def _chain_step(
        self,
        model: SpectralModel,
        j: int,
        g_up: Tensor,
        gate_j: Tensor,
        *,
        exact: bool = False,
    ) -> Tensor:
        """Algorithm 2 line 8: the signal of block ``j`` from that of block
        ``j+1`` through the corrected block ``j+1``.

        ``r = (1/c) V̄ R̄ᵀ g`` splits into the random branch (every non-paired
        column: forward and read directions, random lifts) and the paired
        branch ``Q̃ (ηU)ᵀ g = η Q̃ s``.  Mean-field (``exact=False``) replaces
        the random branch by its per-neuron sample mean; the exact chain keeps
        it per sample.  ``gate_j`` is ``σ'(h̃_j)/√p_j`` on the updated layer.
        """
        blk_up = cast(_Block, model.blocks[j + 1])
        red_up = cast(_Reduce, blk_up.reduce)
        w = cast(Tensor, blk_up.weight)
        tags = self._origin[j + 1]
        pair = torch.tensor(
            [i for i, t in enumerate(tags) if t == PAIR], device=w.device
        )
        non = torch.tensor(
            [i for i, t in enumerate(tags) if t != PAIR], device=w.device
        )
        t = (g_up @ w.transpose(0, 1)) / blk_up.c  # (n, K_up): R̄ᵀ g / c
        rand = t[:, non] @ red_up.V[:, non].transpose(0, 1)  # (n, p_j)
        r_prev = rand if exact else rand.mean(dim=0, keepdim=True).expand_as(rand)
        if pair.numel():
            r_prev = r_prev + t[:, pair] @ red_up.V[:, pair].transpose(0, 1)
        return gate_j * r_prev

    @torch.no_grad()
    def _first_order_step(
        self,
        phase1: SpectralModel,
        j: int,
        g_up: Tensor,
        gate_j: Tensor,
        v_read: Tensor,
        scores: Tensor,
        z_j: Tensor,
        tau_up: Tensor,
    ) -> Tensor:
        """Approx. 6' (first-order backward signal) for block ``j`` from the
        frozen signal ``g_up`` of block ``j+1``:

        ``g^j = gate_j ⊙ [ r̄ + Σ_i ŝ_i q⁰_i / c_{j+1} ]`` with ``r̄`` the
        sample mean of ``(1/c_{j+1}) V_{j+1} R_{j+1}ᵀ g_up`` through the
        Phase-I block ``j+1`` (all columns), ``ŝ`` the frozen scores of
        boundary ``j`` and ``q⁰_i`` the read-direction ``v_i`` lifted through
        the Phase-I block ``j`` (``V_jᵀ v_i`` then ``W_j / c_j``), scaled so
        that ``RMS_μ⟨q⁰_i, z^j_μ⟩`` equals the column step's per-channel
        target ``β √p_{j+1} τ_{j+1}`` (equal amplitude to the installed
        paired columns; only the pattern differs).
        """
        config: RowColConfig = self.config  # type: ignore[assignment]
        chunk = int(config.chunk_size)
        blk_up = cast(_Block, phase1.blocks[j + 1])
        red_up = cast(_Reduce, blk_up.reduce)
        w_up = cast(Tensor, blk_up.weight)
        t = (g_up @ w_up.transpose(0, 1)) / blk_up.c  # (n, K_{j+1}): Rᵀ g / c
        r_bar = (t @ red_up.V.transpose(0, 1)).mean(dim=0, keepdim=True)  # (1, p_j)
        blk = cast(_Block, phase1.blocks[j])
        red = cast(_Reduce, blk.reduce)
        w = cast(Tensor, blk.weight)
        q0 = (v_read.to(red.V.dtype).transpose(0, 1) @ red.V) @ w / blk.c  # (r, p_j)
        p_up = int(w_up.shape[1])
        target = float(config.beta) * math.sqrt(p_up) * float(tau_up)
        rms = _rms_cols(z_j, q0.transpose(0, 1), chunk)  # (r,)
        q0 = q0 * (target / rms.clamp_min(EPS)).to(q0.dtype).unsqueeze(1)
        spike = (scores @ q0) / blk_up.c  # (n, p_j)
        return gate_j * (r_bar + spike)

    def _thin_svd(self, a_mat: Tensor, r: int) -> tuple[Tensor, Tensor, Tensor]:
        """Top-``r`` singular triplets of ``a_mat`` (randomized when it pays)."""
        config: RowColConfig = self.config  # type: ignore[assignment]
        q = min(r + int(config.svd_oversample), min(a_mat.shape))
        if q >= min(a_mat.shape) // 2:
            u, s, vh = torch.linalg.svd(a_mat, full_matrices=False)
            return u[:, :r], s[:r], vh[:r].transpose(0, 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config.lift_seed))
            u, s, v = torch.svd_lowrank(a_mat, q=q, niter=int(config.svd_niter))
        return u[:, :r], s[:r], v[:, :r]

    def _slots(
        self,
        model: SpectralModel,
        j: int,
        v: Tensor,
        z: Tensor,
        y: Tensor,
        r: int,
        chunk: int,
        tag: str,
    ) -> tuple[Tensor, Tensor]:
        """Indices of the ``r`` columns a swap of block ``j`` removes, and the
        mean coordinate RMS of the survivors (the forward-coordinate scale)."""
        config: RowColConfig = self.config  # type: ignore[assignment]
        tags = self._origin[j]
        if config.swap_rule == "importance":
            score = self._importance(model, j, v, z, y, chunk)
        else:
            score = _rayleigh(z, y, v, chunk)
        if config.swap_rule == "replace_own":
            own = [i for i, t in enumerate(tags) if t == tag]
            if len(own) >= r:
                drop = torch.tensor(own[:r], device=v.device)
            else:
                fwd = torch.tensor(
                    [i for i, t in enumerate(tags) if t == FWD], device=v.device
                )
                drop = torch.cat(
                    [
                        torch.tensor(own, device=v.device, dtype=torch.long),
                        fwd[torch.argsort(score[fwd])[: r - len(own)]],
                    ]
                )
        else:
            drop = torch.argsort(score)[:r]
        keep = torch.ones(v.shape[1], dtype=torch.bool, device=v.device)
        keep[drop] = False
        return drop, self._tau(j, v, z, keep, chunk)

    def _tau(self, j: int, v: Tensor, z: Tensor, keep: Tensor, chunk: int) -> Tensor:
        """Forward-coordinate scale of block ``j``: the mean coordinate RMS of
        the kept columns that ``tau_over`` admits."""
        config: RowColConfig = self.config  # type: ignore[assignment]
        if config.tau_over == "random_lift":
            rl = torch.tensor(
                [t in (FWD, READ) for t in self._origin[j]],
                dtype=torch.bool,
                device=v.device,
            )
            if bool((keep & rl).any()):
                keep = keep & rl
        return _rms_cols(z, v[:, keep], chunk).mean()

    # ------------------------------------------------------------------ #
    # Algorithm 3: update of block ℓ+1
    # ------------------------------------------------------------------ #

    def _init_block_state(self, model: SpectralModel) -> None:
        """Bookkeeping of ``upper_update="block"``: the ghosts of every block
        (forward columns swapped out: alignment targets, never installed), the
        scores of every boundary of every pass, and for every read / paired
        slot the scores that define it."""
        config: RowColConfig = self.config  # type: ignore[assignment]
        self._block_mode = config.upper_update == "block"
        self._pass = 0
        self._ghost: list[Tensor] = []
        for b in model.blocks:
            red = cast(_Reduce, cast(_Block, b).reduce)
            if red.is_identity:  # nothing to refit (e.g. the CNN's conv1)
                self._ghost.append(torch.zeros(0, 0, dtype=torch.float64))
                continue
            v = _reduce_cols(red)
            self._ghost.append(v.new_zeros(v.shape[0], 0, dtype=torch.float64))
        self._slot_src: list[dict[int, tuple[tuple[int, int], int]]] = [
            {} for _ in model.blocks
        ]
        self._score_store: dict[tuple[int, int], Tensor] = {}

    def _capture_ghosts(self, j: int, v: Tensor, drop: Tensor) -> None:
        """Remember the forward columns of block ``j`` about to be overwritten
        (``v``: ``(p, K)`` current columns)."""
        if not self._block_mode:
            return
        idx = [i for i in drop.tolist() if self._origin[j][i] == FWD]
        if idx:
            self._ghost[j] = torch.cat([self._ghost[j], v[:, idx].double()], dim=1)

    def _register(
        self, j: int, slots: Tensor, scores: Tensor, *, boundary: int
    ) -> None:
        """Slots ``slots`` of block ``j`` now hold ``E[y s_i z]`` of the scores
        of ``boundary`` in this pass; scores no surviving slot refers to are
        released."""
        if not self._block_mode:
            return
        key = (self._pass, boundary)
        self._score_store[key] = scores
        for i, slot in enumerate(slots.tolist()):
            self._slot_src[j][slot] = (key, i)
        live = {k for src in self._slot_src for k, _ in src.values()}
        for k in [k for k in self._score_store if k not in live]:
            del self._score_store[k]

    @torch.no_grad()
    def _update_block(
        self, model: SpectralModel, b: int, z: Tensor, y: Tensor, chunk: int
    ) -> dict[str, float]:
        """Algorithm 3 on the FFN block ``b`` whose input is now ``z``."""
        config: RowColConfig = self.config  # type: ignore[assignment]
        blk = cast(_Block, model.blocks[b])
        red = cast(_Reduce, blk.reduce)
        tags = self._origin[b]
        out: dict[str, float] = {}
        surv = [i for i, t in enumerate(tags) if t == FWD]
        if surv:
            c_y, m_y, sigma = _second_moments(z, y, chunk)
            old = torch.cat([red.V[:, surv].double(), self._ghost[b]], dim=1)
            rcfg = model.block_config.blocks[b].reduce
            new, out = refit_aligned(c_y, m_y, sigma, old, rcfg)
            moved = (new[:, : len(surv)] - old[:, : len(surv)]).pow(2).sum()
            out["moved"] = float(moved) / len(surv)
            red.V[:, surv] = new[:, : len(surv)].to(red.V.dtype)
            self._ghost[b] = new[:, len(surv) :]
        out["n_forward"] = float(len(surv))
        src = self._slot_src[b]
        slots = [i for i, t in enumerate(tags) if t != FWD and i in src]
        out["n_refreshed"] = float(len(slots))
        if not slots:
            return out
        keep = torch.ones(red.V.shape[1], dtype=torch.bool, device=red.V.device)
        tau = float(self._tau(b, red.V, z, keep, chunk))
        p_b = int(cast(Tensor, blk.weight).shape[1])
        ones = y.new_ones(y.shape)
        for key in {src[i][0] for i in slots}:
            mine = [i for i in slots if src[i][0] == key]
            s = self._score_store[key][:, [src[i][1] for i in mine]]
            cols = _cross(z, y.unsqueeze(1) * s, ones, chunk)  # E[z (y s)ᵀ]
            rms = _rms_cols(z, cols.to(z.dtype), chunk).clamp_min(EPS)
            target = torch.tensor(
                [
                    config.beta * tau * (math.sqrt(p_b) if tags[i] == PAIR else 1.0)
                    for i in mine
                ],
                dtype=torch.float64,
                device=cols.device,
            )
            red.V[:, mine] = (cols * (target / rms).unsqueeze(0)).to(red.V.dtype)
        return out

    @torch.no_grad()
    def _importance(
        self, model: SpectralModel, j: int, v: Tensor, z: Tensor, y: Tensor, chunk: int
    ) -> Tensor:
        """Gradient importance of every column of block ``j``'s reduce
        (section "Feature importance at fixed budget", exact ``Ω``).

        For column ``a`` with unit direction ``v̂_a`` and coordinate
        ``h_a = ⟨v̂_a, z⟩``, the per-neuron amplification rate of the
        neuron's coefficient ``c_{j,a}`` on ``v̂_a`` is (eq:cdot, Approx.
        "Diagonal driver") ``e λ_j(a) c_{j,a} + d ℓ_j(a)`` with
        ``λ_j(a) = ⟨ω_j, Λ₂(a)⟩``, ``ℓ_j(a) = ⟨ω_j, Λ₁(a)⟩``,
        ``Λ₂(a) = E[y ĝ h_a²]``, ``Λ₁(a) = E[y ĝ h_a]``, ``ω_j`` the neuron's
        output column in the block above, ``ĝ = ḡ + UUᵀ(g − ḡ)`` the retained
        signal of that boundary and ``(d, e) = (E[σ'(h)], E[hσ'(h)])`` of this
        layer.  The importance is the RMS of the rate over the neurons.  For
        the top block the layer above is the readout: ``ω_j = a_j``, ``ĝ ≡ 1``.
        """
        n = z.shape[0]
        blk = cast(_Block, model.blocks[j])
        norms = v.norm(dim=0).clamp_min(EPS)
        vn = v / norms
        d_l, e_l = self._gate_coeffs(blk, z, chunk)
        # lift coefficient of neuron i on v̂_a: weight[a, i]·‖v_a‖ / c
        coef = (cast(Tensor, blk.weight) * norms.unsqueeze(1)).transpose(0, 1) / blk.c
        top = j == len(model.blocks) - 1
        if top:
            lam = torch.zeros(v.shape[1], dtype=torch.float64, device=z.device)
            mom = torch.zeros_like(lam)
            for zc, yc in zip(z.split(chunk), y.split(chunk), strict=True):
                h = (zc @ vn).double()
                yd = yc.double().unsqueeze(1)
                lam += (yd * h * h).sum(dim=0)
                mom += (yd * h).sum(dim=0)
            lam, mom = lam / n, mom / n
            a_out = self._readout_grad(model).double().unsqueeze(1)  # (p_j, 1)
            rate = a_out * (
                e_l * lam.unsqueeze(0) * coef.double() + d_l * mom.unsqueeze(0)
            )
            return rate.pow(2).mean(dim=0)
        g_up, u = self._ctx[j]  # signal of block j+1 and write-directions
        g_bar = g_up.mean(dim=0)  # (p_{j+1},)
        s_c = (g_up - g_bar.unsqueeze(0)) @ u  # centred retained scores (n, r)
        k_cols = v.shape[1]
        lam = torch.zeros(k_cols, dtype=torch.float64, device=z.device)
        mom = torch.zeros_like(lam)
        lam_s = torch.zeros(u.shape[1], k_cols, dtype=torch.float64, device=z.device)
        mom_s = torch.zeros_like(lam_s)
        for zc, yc, sc in zip(
            z.split(chunk), y.split(chunk), s_c.split(chunk), strict=True
        ):
            h = (zc @ vn).double()
            yd = yc.double().unsqueeze(1)
            lam += (yd * h * h).sum(dim=0)
            mom += (yd * h).sum(dim=0)
            ys_c = (yd * sc.double()).transpose(0, 1)  # (r, n_c)
            lam_s += ys_c @ (h * h)
            mom_s += ys_c @ h
        lam, mom, lam_s, mom_s = lam / n, mom / n, lam_s / n, mom_s / n
        # projections of Λ₂(a), Λ₁(a) on every output column ω_i = V̄[i,:] R̄ᵀ / c
        blk_up = cast(_Block, model.blocks[j + 1])
        r_bar = cast(Tensor, blk_up.weight).double()  # (K_up, p_{j+1}) = R̄ᵀ
        v_bar = cast(_Reduce, blk_up.reduce).V.double()  # (p_j, K_up)
        rg = r_bar @ g_bar.double()  # (K_up,)
        ru = r_bar @ u.double()  # (K_up, r)
        t2 = (
            v_bar
            @ (rg.unsqueeze(1) * lam.unsqueeze(0) + ru @ lam_s)
            / blk_up.c.double()
        )
        t1 = (
            v_bar
            @ (rg.unsqueeze(1) * mom.unsqueeze(0) + ru @ mom_s)
            / blk_up.c.double()
        )
        rate = e_l * t2 * coef.double() + d_l * t1  # (p_j, K)
        return rate.pow(2).mean(dim=0)

    @torch.no_grad()
    def _gate_coeffs(
        self, blk: _Block, a_in: Tensor, chunk: int
    ) -> tuple[float, float]:
        """``(d, e) = (E[σ'(h)], E[h σ'(h)])`` of the block on its current input."""
        transform = cast(Any, blk.transform)
        s1 = s2 = 0.0
        count = 0
        for ac in a_in.split(chunk):
            h = transform.project(blk.expand_input(ac)) / blk.c
            gp = _sigma_prime(h, str(blk.activation))
            s1 += float(gp.double().sum())
            s2 += float((h.double() * gp.double()).sum())
            count += h.numel()
        return s1 / max(count, 1), s2 / max(count, 1)

    def _retag(self, j: int, idx: Tensor, tag: str) -> dict[str, int]:
        tags = self._origin[j]
        dropped = {o: 0 for o in (FWD, READ, PAIR)}
        for i in idx.tolist():
            dropped[tags[i]] += 1
            tags[i] = tag
        return dropped

    @torch.no_grad()
    def _block_state(
        self, blk: _Block, a_in: Tensor, chunk: int
    ) -> tuple[Tensor, Tensor]:
        outs, gts = [], []
        for ac in a_in.split(chunk):
            gts.append(block_derivative_features(blk, ac))
            outs.append(blk(ac))
        return torch.cat(outs), torch.cat(gts)

    @torch.no_grad()
    def _state(
        self, model: SpectralModel, X: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        chunk = int(self.config.chunk_size)  # type: ignore[attr-defined]
        feats: list[Tensor] = []
        gates: list[Tensor] = []
        a = X
        for blk in model.blocks:
            b = cast(_Block, blk)
            z, g = self._block_state(b, a, chunk)
            feats.append(z)
            gates.append(g)
            a = z
        return feats, gates

    @staticmethod
    def _materialise(loader: DataLoader, dev: torch.device) -> tuple[Tensor, Tensor]:
        xs, ys = [], []
        for x, yb in loader:
            xs.append(x.to(dev, non_blocking=True))
            yb = yb.to(dev, non_blocking=True).to(torch.float32)
            ys.append(yb.reshape(yb.shape[0]))
        return torch.cat(xs), torch.cat(ys)
