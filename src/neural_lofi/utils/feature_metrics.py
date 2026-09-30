"""Pluggable feature-comparison metrics for two aligned representations.

Every metric compares a *reference* feature batch ``ref`` against a *comparison*
batch ``cmp`` — the per-layer outputs of two models evaluated on the **same**
samples in the **same order** (e.g. a spectral NLoFi CNN vs. its GD-trained
twin).  Supported input shapes:

* ``(N, C, H, W)`` — convolutional features.  Channels are the feature axis;
  the ``H*W`` spatial positions are treated as independent *locations* and the
  per-location metric is averaged (the same convention used by CKA and the
  correlation-map metrics).
* ``(N, D)`` — flattened / fully-connected features (a single location).

The two batches may have **different channel counts** ``C_ref != C_cmp``; every
metric here is well-defined in that case.

Each metric is a callable ``metric(ref, cmp, **params) -> dict[str, ReductionValue]``
returning named scalars (and occasionally short vectors).  The registry
:data:`METRICS` maps a short name to its callable; :func:`flatten_metric_dict`
expands any vector entries into indexed scalar columns (``name``, ``name_0``,
``name_1``, …) for tabular storage.

The metrics live in the package (rather than a one-off script) so the spectral
vs. GD feature-comparison study and any later analysis share one tested
implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
from torch import Tensor

ReductionValue = float | list[float]

# A metric maps an aligned (ref, cmp) feature pair to named summary values.
FeatureMetric = Callable[..., dict[str, ReductionValue]]


# --------------------------------------------------------------------------- #
# Shared shape handling
# --------------------------------------------------------------------------- #
def _to_nlc(feat: Tensor) -> Tensor:
    """Return ``feat`` reshaped to ``(N, L, C)`` (locations × channels).

    ``(N, C, H, W) -> (N, H*W, C)`` (channels last); ``(N, D) -> (N, 1, D)``.
    """
    if feat.ndim == 4:
        return feat.permute(0, 2, 3, 1).reshape(feat.shape[0], -1, feat.shape[1])
    if feat.ndim == 2:
        return feat.unsqueeze(1)
    raise ValueError(f"Unsupported feature shape {tuple(feat.shape)}; expected 4D/2D.")


def _pair_to_nlc(ref: Tensor, cmp: Tensor) -> tuple[Tensor, Tensor]:
    """Validate a ``(ref, cmp)`` pair and return both as CPU float32 ``(N, L, C)``.

    Both batches must share the sample count ``N`` and rank; for 4D inputs the
    spatial shape (hence location count ``L``) must match.  Channel counts may
    differ.
    """
    if ref.shape[0] != cmp.shape[0]:
        raise ValueError(f"Sample-count mismatch: {ref.shape[0]} != {cmp.shape[0]}")
    if ref.ndim != cmp.ndim:
        raise ValueError(
            f"Feature-rank mismatch: {tuple(ref.shape)} vs {tuple(cmp.shape)}"
        )
    if ref.ndim == 4 and ref.shape[2:] != cmp.shape[2:]:
        raise ValueError(
            f"Spatial-shape mismatch: {tuple(ref.shape[2:])} vs {tuple(cmp.shape[2:])}"
        )

    ref_nlc = _to_nlc(ref).to(device="cpu", dtype=torch.float32)
    cmp_nlc = _to_nlc(cmp).to(device="cpu", dtype=torch.float32)

    n = int(ref_nlc.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 samples to compute feature metrics")
    if ref_nlc.shape[1] != cmp_nlc.shape[1]:
        raise ValueError(
            f"Location-count mismatch: {ref_nlc.shape[1]} vs {cmp_nlc.shape[1]}"
        )
    return ref_nlc, cmp_nlc


# --------------------------------------------------------------------------- #
# Pearson correlation map + reductions  (ported from compare_cnn_features.py)
# --------------------------------------------------------------------------- #
def _feature_correlation_map(ref_nlc: Tensor, cmp_nlc: Tensor, *, eps: float) -> Tensor:
    """Per-location channel×channel Pearson correlation, shape ``(L, C_ref, C_cmp)``."""
    n, length, c_ref = ref_nlc.shape
    c_cmp = cmp_nlc.shape[2]
    corr_map = torch.empty((length, c_ref, c_cmp), dtype=torch.float32)
    for loc in range(length):
        x = ref_nlc[:, loc, :]
        y = cmp_nlc[:, loc, :]
        x_c = x - x.mean(dim=0, keepdim=True)
        y_c = y - y.mean(dim=0, keepdim=True)
        cov = x_c.T @ y_c
        x_norm = torch.sqrt((x_c * x_c).sum(dim=0).clamp_min(eps))
        y_norm = torch.sqrt((y_c * y_c).sum(dim=0).clamp_min(eps))
        denom = (x_norm.unsqueeze(1) * y_norm.unsqueeze(0)).clamp_min(eps)
        corr_map[loc] = cov / denom
    return corr_map


def _corr_map_reductions(corr_map: Tensor, *, prefix: str) -> dict[str, ReductionValue]:
    """Scalar/vector reductions of a ``(L, C_ref, C_cmp)`` correlation-style map.

    ``prefix="corr"`` reproduces the legacy column names used in the paper
    (``corr_mean``, ``corr_top3_eigenvec``, ``overlap_norm_fro``); other
    prefixes namespace the same reductions (e.g. ``dcor_*``).
    """
    overlap_key = (
        "overlap_norm_fro" if prefix == "corr" else f"{prefix}_overlap_norm_fro"
    )
    return {
        f"{prefix}_mean": float(corr_map.mean().item()),
        f"{prefix}_max": float(corr_map.max().item()),
        f"{prefix}_min": float(corr_map.min().item()),
        f"{prefix}_top3_eigenvec": (corr_map[:, :3, :] ** 2)
        .mean(dim=(0, 2))
        .flatten()
        .cpu()
        .tolist(),
        overlap_key: float(
            (torch.linalg.norm(corr_map) / (corr_map.numel() ** 0.5)).item()
        ),
    }


def pearson_metric(
    ref: Tensor, cmp: Tensor, *, eps: float = 1e-8
) -> dict[str, ReductionValue]:
    """Pearson channel-correlation map reduced to summary scalars."""
    ref_nlc, cmp_nlc = _pair_to_nlc(ref, cmp)
    corr_map = _feature_correlation_map(ref_nlc, cmp_nlc, eps=eps)
    return _corr_map_reductions(corr_map, prefix="corr")


# --------------------------------------------------------------------------- #
# Linear CKA (per-location channel features, averaged over locations)
# --------------------------------------------------------------------------- #
def cka_metric(
    ref: Tensor, cmp: Tensor, *, eps: float = 1e-8
) -> dict[str, ReductionValue]:
    """Linear CKA between channel features at each location, averaged over locations.

    ``CKA_l = ||S_xy||_F^2 / (||S_xx||_F ||S_yy||_F)`` with sample-covariance
    operators; well-defined when ``C_ref != C_cmp``.
    """
    ref_nlc, cmp_nlc = _pair_to_nlc(ref, cmp)
    n, length, _ = ref_nlc.shape
    scale = 1.0 / max(n - 1, 1)

    cka = torch.empty(length, dtype=torch.float32)
    for loc in range(length):
        x = ref_nlc[:, loc, :]
        y = cmp_nlc[:, loc, :]
        x_c = x - x.mean(dim=0, keepdim=True)
        y_c = y - y.mean(dim=0, keepdim=True)
        s_xy = (x_c.T @ y_c) * scale
        s_xx = (x_c.T @ x_c) * scale
        s_yy = (y_c.T @ y_c) * scale
        num = (s_xy * s_xy).sum()
        den_x = torch.sqrt((s_xx * s_xx).sum().clamp_min(eps))
        den_y = torch.sqrt((s_yy * s_yy).sum().clamp_min(eps))
        cka[loc] = num / (den_x * den_y).clamp_min(eps)

    return {
        "cka_channel_final": float(cka.mean().item()),
        "cka_loc_mean": float(cka.mean().item()),
        "cka_loc_max": float(cka.max().item()),
        "cka_loc_min": float(cka.min().item()),
    }


# --------------------------------------------------------------------------- #
# Distance correlation map (exact, chunked over channels)
# --------------------------------------------------------------------------- #
def _centered_abs_dist(values: Tensor) -> Tensor:
    """Double-centered pairwise absolute distances, shape ``(N, N, C)``."""
    dist = torch.abs(values.unsqueeze(1) - values.unsqueeze(0))
    row = dist.mean(dim=1, keepdim=True)
    col = dist.mean(dim=0, keepdim=True)
    grand = dist.mean(dim=(0, 1), keepdim=True)
    return dist - row - col + grand


def _feature_dcorr_map(
    ref_nlc: Tensor, cmp_nlc: Tensor, *, eps: float, channel_chunk_size: int
) -> Tensor:
    """Per-location channel×channel distance correlation, shape ``(L, C_ref, C_cmp)``.

    Peak memory is bounded by chunking channels and processing one location at a
    time (``~2 * N^2 * channel_chunk_size * 4`` bytes).  Runs on the inputs'
    device (see ``dcor_metric``'s ``device``).
    """
    n, length, c_ref = ref_nlc.shape
    c_cmp = cmp_nlc.shape[2]
    dev = ref_nlc.device
    total_cov = torch.zeros((length, c_ref, c_cmp), dtype=torch.float32, device=dev)
    var_ref = torch.zeros((length, c_ref), dtype=torch.float32, device=dev)
    var_cmp = torch.zeros((length, c_cmp), dtype=torch.float32, device=dev)
    cr_bs = min(c_ref, channel_chunk_size)
    cc_bs = min(c_cmp, channel_chunk_size)
    nn = n * n

    for loc in range(length):
        ref_loc = ref_nlc[:, loc, :]
        cmp_loc = cmp_nlc[:, loc, :]
        for cr_s in range(0, c_ref, cr_bs):
            cr_e = min(cr_s + cr_bs, c_ref)
            ref_flat = _centered_abs_dist(ref_loc[:, cr_s:cr_e]).reshape(
                nn, cr_e - cr_s
            )
            var_ref[loc, cr_s:cr_e] += (ref_flat * ref_flat).sum(dim=0)
            for cc_s in range(0, c_cmp, cc_bs):
                cc_e = min(cc_s + cc_bs, c_cmp)
                cmp_flat = _centered_abs_dist(cmp_loc[:, cc_s:cc_e]).reshape(
                    nn, cc_e - cc_s
                )
                if cr_s == 0:
                    var_cmp[loc, cc_s:cc_e] += (cmp_flat * cmp_flat).sum(dim=0)
                total_cov[loc, cr_s:cr_e, cc_s:cc_e] += ref_flat.T @ cmp_flat

    denom = torch.sqrt(var_ref.unsqueeze(-1) * var_cmp.unsqueeze(-2)).clamp_min(eps)
    return total_cov / denom


def dcor_metric(
    ref: Tensor,
    cmp: Tensor,
    *,
    eps: float = 1e-8,
    channel_chunk_size: int = 16,
    device: str | None = None,
) -> dict[str, ReductionValue]:
    """Distance-correlation map reduced to summary scalars (prefix ``dcor_``).

    Exact but ``O(N^2)`` in memory/compute — opt-in for large ``N``.  ``device``
    moves the computation (e.g. ``"cuda"``; ~5x faster at the probe caps); the
    result is identical up to float reduction order.
    """
    ref_nlc, cmp_nlc = _pair_to_nlc(ref, cmp)
    if device is not None:
        ref_nlc, cmp_nlc = ref_nlc.to(device), cmp_nlc.to(device)
    dcorr_map = _feature_dcorr_map(
        ref_nlc, cmp_nlc, eps=eps, channel_chunk_size=channel_chunk_size
    )
    return _corr_map_reductions(dcorr_map.cpu(), prefix="dcor")


# --------------------------------------------------------------------------- #
# Subspace-alignment metric (symmetric top-K PCA, sample-space)
# --------------------------------------------------------------------------- #
def subspace_metric(
    ref: Tensor,
    cmp: Tensor,
    *,
    dims: Sequence[int] = (1, 5, 10, 25),
    center: bool = True,
    eps: float = 1e-8,
) -> dict[str, ReductionValue]:
    """Principal-angle overlap of the two top-K feature subspaces, per fixed K.

    For each location and each ``K`` in ``dims`` the overlap is

    ``overlap_K = ||U_ref[:, :K]^T U_cmp[:, :K]||_F^2 / K``,

    where ``U_ref`` / ``U_cmp`` are the top-K **left** singular vectors of the
    (optionally) sample-centered per-location feature matrices.  Working in the
    shared **sample space** ``R^N`` makes the comparison well-defined even when
    ``C_ref != C_cmp``.  The value is the mean squared cosine of the principal
    angles between the two K-dimensional subspaces: it is bounded in ``[0, 1]``
    and equals its maximum ``1`` exactly when the subspaces coincide (``0`` when
    orthogonal).  Overlaps are averaged over spatial locations.

    Returns one entry ``subspace_overlap_k{K}`` per requested ``K``.  ``K`` is
    capped per call at ``min(C_ref, C_cmp, N)``; requests above the cap reuse the
    cap's subspace.
    """
    ref_nlc, cmp_nlc = _pair_to_nlc(ref, cmp)
    n, length, c_ref = ref_nlc.shape
    c_cmp = cmp_nlc.shape[2]

    requested = sorted({int(k) for k in dims if int(k) >= 1})
    if not requested:
        raise ValueError(f"subspace dims must contain a positive integer, got {dims!r}")
    k_cap = min(c_ref, c_cmp, n)
    k_max = min(max(requested), k_cap)

    xr = ref_nlc.permute(1, 0, 2)  # (L, N, C_ref)
    xc = cmp_nlc.permute(1, 0, 2)  # (L, N, C_cmp)
    if center:
        xr = xr - xr.mean(dim=1, keepdim=True)
        xc = xc - xc.mean(dim=1, keepdim=True)

    # Top-k_max left singular vectors per location, orthonormal columns in R^N.
    u_ref = torch.linalg.svd(xr, full_matrices=False).U[:, :, :k_max]  # (L, N, k_max)
    u_cmp = torch.linalg.svd(xc, full_matrices=False).U[:, :, :k_max]

    cross = u_ref.transpose(1, 2) @ u_cmp  # (L, k_max, k_max)
    sq = cross * cross
    # cumsq[:, a, b] = sum over the top-(a+1) ref × top-(b+1) cmp block.
    cumsq = sq.cumsum(1).cumsum(2)
    diag_cumsq = torch.diagonal(cumsq, dim1=1, dim2=2)  # (L, k_max)

    out: dict[str, ReductionValue] = {}
    for k in requested:
        kk = min(k, k_max)
        overlap = (diag_cumsq[:, kk - 1] / kk).mean().clamp(0.0, 1.0)
        out[f"subspace_overlap_k{k}"] = float(overlap.item())
    return out


# --------------------------------------------------------------------------- #
# Directed subspace-recovery metric (canonical correlations of feature spans)
# --------------------------------------------------------------------------- #
def _as_sample_matrix(feat: Tensor) -> Tensor:
    """Return ``feat`` as a 2-D sample×feature matrix, locations as samples.

    ``(N, C, H, W) -> (N*H*W, C)`` (every spatial position is a sample);
    ``(N, D)`` passes through.  This is the layout the recovery metric works in
    — spans live in channel space over all samples *and* locations.
    """
    if feat.ndim == 4:
        return feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
    if feat.ndim == 2:
        return feat
    raise ValueError(f"Unsupported feature shape {tuple(feat.shape)}; expected 4D/2D.")


def _orthonormal_span(features: Tensor, *, eps: float) -> Tensor:
    """Orthonormal basis (in sample space) of the centered feature span.

    Columns of ``U`` from the compact SVD of the sample-centered matrix, kept up
    to numerical rank — invariant to feature scaling, signs, and rotations.
    Large matrices run the SVD on CUDA when available (the metric itself is
    called with CPU tensors; only the decomposition is offloaded).
    """
    centered = features.to(dtype=torch.float32)
    centered = centered - centered.mean(dim=0, keepdim=True)
    device = (
        "cuda"
        if centered.numel() > 1_000_000 and torch.cuda.is_available()
        else centered.device
    )
    u, s, _ = torch.linalg.svd(centered.to(device), full_matrices=False)
    if s.numel() == 0:
        return centered.new_empty(centered.shape[0], 0)
    tolerance = eps * max(centered.shape) * s.max().clamp_min(eps)
    rank = int((s > tolerance).sum().item())
    basis: Tensor = u[:, :rank].to(centered.device).contiguous()
    return basis


def recovery_metric(
    ref: Tensor,
    cmp: Tensor,
    *,
    eps: float = 1e-8,
    max_samples: int = 8192,
    seed: int = 0,
) -> dict[str, ReductionValue]:
    """Directed recovery of the ``cmp`` feature span by the ``ref`` span.

    Convention (from the FC LoFi subspace-recovery study): ``ref`` = spectral
    (LoFi) features, ``cmp`` = GD features.  With ``Q_s`` / ``Q_g`` orthonormal
    bases of the centered spans over the same samples, the headline score is

        ``subspace_recovery_gd_by_spectral = ||Q_s^T Q_g||_F^2 / rank(Q_g)``

    — the mean squared canonical correlation of the GD span with the LoFi span;
    1 exactly when the GD span is contained in the LoFi span.  ``precision`` is
    the reverse normalization (by ``rank(Q_s)``).  Conv features are compared
    locations-as-samples (``(N, C, H, W) -> (N*H*W, C)``); when the sample count
    exceeds ``max_samples`` the *same* seeded row subset is taken on both sides
    (the bases need ``O(S x C)`` memory).
    """
    if ref.shape[0] != cmp.shape[0]:
        raise ValueError(f"Sample-count mismatch: {ref.shape[0]} != {cmp.shape[0]}")
    if ref.ndim == 4 and cmp.ndim == 4 and ref.shape[2:] != cmp.shape[2:]:
        raise ValueError(
            f"Spatial-shape mismatch: {tuple(ref.shape[2:])} vs {tuple(cmp.shape[2:])}"
        )
    x = _as_sample_matrix(ref).to(device="cpu", dtype=torch.float32)
    y = _as_sample_matrix(cmp).to(device="cpu", dtype=torch.float32)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Row-count mismatch: {x.shape[0]} != {y.shape[0]}")
    if x.shape[0] < 2:
        raise ValueError("Need at least two samples for subspace recovery")
    if max_samples > 0 and x.shape[0] > max_samples:
        gen = torch.Generator().manual_seed(seed)
        rows = torch.randperm(x.shape[0], generator=gen)[:max_samples]
        x, y = x[rows], y[rows]

    q_spectral = _orthonormal_span(x, eps=eps)
    q_gd = _orthonormal_span(y, eps=eps)
    spectral_rank = int(q_spectral.shape[1])
    gd_rank = int(q_gd.shape[1])
    if spectral_rank == 0 or gd_rank == 0:
        return {
            "subspace_rank_spectral": float(spectral_rank),
            "subspace_rank_gd": float(gd_rank),
            "subspace_recovery_gd_by_spectral": 0.0,
            "subspace_precision_spectral_in_gd": 0.0,
            "subspace_mean_canonical_corr": 0.0,
            "subspace_max_canonical_corr": 0.0,
        }

    overlap = q_spectral.T @ q_gd
    canonical_corr = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)
    fro_sq = overlap.pow(2).sum()
    return {
        "subspace_rank_spectral": float(spectral_rank),
        "subspace_rank_gd": float(gd_rank),
        "subspace_recovery_gd_by_spectral": float((fro_sq / gd_rank).item()),
        "subspace_precision_spectral_in_gd": float((fro_sq / spectral_rank).item()),
        "subspace_mean_canonical_corr": float(canonical_corr.mean().item()),
        "subspace_max_canonical_corr": float(canonical_corr.max().item()),
    }


# --------------------------------------------------------------------------- #
# Registry + flattening
# --------------------------------------------------------------------------- #
METRICS: dict[str, FeatureMetric] = {
    "pearson": pearson_metric,
    "cka": cka_metric,
    "dcor": dcor_metric,
    "subspace": subspace_metric,
    "recovery": recovery_metric,
}


def flatten_metric_dict(reductions: dict[str, ReductionValue]) -> dict[str, float]:
    """Expand vector reductions into indexed scalar columns for tabular storage."""
    out: dict[str, float] = {}
    for key, value in reductions.items():
        if isinstance(value, list):
            for i, component in enumerate(value):
                out[f"{key}_{i}"] = float(component)
        else:
            out[key] = float(value)
    return out


def compute_metrics(
    ref: Tensor,
    cmp: Tensor,
    names: Iterable[str],
    *,
    params: dict[str, dict[str, object]] | None = None,
) -> dict[str, float]:
    """Run the named metrics on one ``(ref, cmp)`` pair and return flat scalars.

    ``params`` optionally maps a metric name to keyword overrides (e.g.
    ``{"subspace": {"dims": [1, 5, 10, 25]}}``).
    """
    params = params or {}
    row: dict[str, float] = {}
    for name in names:
        try:
            metric = METRICS[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown feature metric {name!r}; available: {sorted(METRICS)}"
            ) from exc
        row.update(flatten_metric_dict(metric(ref, cmp, **params.get(name, {}))))
    return row
