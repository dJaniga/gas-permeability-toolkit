"""Pulse-decay (pressure-decay) permeability.

A core plug sits between two **closed** vessels, upstream ``V1`` and downstream
``V2``, both at pore pressure ``P_mean``. A small pulse ``dP0`` is applied to
``V1``; the differential pressure decays through the plug as ``dP(t) =
dP0 * exp(-alpha * t)``, and permeability follows from the decay *rate*.

**No flow is measured.** That is the whole point: below roughly ten microdarcy
a thermal mass flowmeter sized for a normal plug sits at a fraction of a percent
of full scale and reports its own zero offset, which is stable enough to pass
every steady-state check while being no measurement at all. Pulse decay removes
that instrument from the measurement entirely, which is why it is the standard
method down here.

Two models, both implemented:

**Zero storage (Brace et al. 1968)** -- the sample's own pore volume is
negligible against the vessels::

    alpha = k*A / (mu*c_g*L) * (1/V1 + 1/V2)

**Sample storage (Dicker & Smits 1988)** -- it is not, which is the usual case
once the vessels are small enough to give a workable run time. With
``V_p = phi*A*L``, ``a1 = V_p/V1`` and ``a2 = V_p/V2``::

    alpha = theta_1^2 * k / (phi*mu*c_g*L^2)

where ``theta_1`` is the first root of the storage equation below. As the
storage ratios go to zero this reduces exactly to the Brace form.

**Units.** Everything here is strict CGS-Darcy -- k in darcy, A in cm^2, mu in
cP, L in cm, V in cm^3, and gas compressibility in **1/atm** -- in which the
expression above yields alpha in 1/s with no conversion constant at all. That
is not a coincidence: the darcy is defined as ``cP*cm^2/(s*atm)``, so the
reciprocal-pressure compressibility cancels the atm in the darcy. Plain floats
in, plain floats out, no config objects, so every function here is testable
against hand-worked numbers with no hardware and no configuration.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from gasperm.config.run import PulseDecayConfig
from gasperm.models import DecayFit, PulseDecayStatus

__all__ = [
    "PulseDecayInputError",
    "pore_volume_cm3",
    "storage_ratios",
    "first_storage_root",
    "brace_permeability_darcy",
    "dicker_smits_permeability_darcy",
    "brace_decay_rate_per_s",
    "dicker_smits_decay_rate_per_s",
    "find_pulse",
    "fit_window",
    "fit_decay_rate",
    "PulseDecayMonitor",
]

#: Newton/brentq tolerance for the storage root. theta_1 enters k squared, so
#: 1e-12 in the root is 2e-12 in the result -- far below every other term.
_ROOT_TOLERANCE = 1.0e-12

#: Smallest storage ratio that is worth solving for. Below this the pole-free
#: residual is dominated by floating-point noise and the Brace limit is exact
#: to well past double precision anyway.
_NEGLIGIBLE_STORAGE = 1.0e-12


class PulseDecayInputError(ValueError):
    """A pulse-decay input is missing, non-physical, or self-contradictory."""


def _require_positive(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise PulseDecayInputError(f"{name} must be a finite number, got {value!r}.")
    if value <= 0.0:
        raise PulseDecayInputError(f"{name} must be positive, got {value!r}.")
    return value


# --------------------------------------------------------------------------
# Storage geometry
# --------------------------------------------------------------------------


def pore_volume_cm3(
    *, area_cm2: float, length_cm: float, porosity_fraction: float
) -> float:
    """Connected pore volume of the plug, cm^3.

    This is the volume that has to be filled before the plug can pass gas
    through, which is exactly why it competes with the vessels.
    """
    _require_positive("area_cm2", area_cm2)
    _require_positive("length_cm", length_cm)
    if not 0.0 <= porosity_fraction <= 1.0:
        raise PulseDecayInputError(
            f"porosity_fraction must be between 0 and 1, got {porosity_fraction!r}."
        )
    return porosity_fraction * area_cm2 * length_cm


def storage_ratios(
    *,
    pore_volume_cm3: float,
    upstream_volume_cm3: float,
    downstream_volume_cm3: float,
) -> tuple[float, float]:
    """``(a1, a2)`` -- pore volume against each vessel.

    These are the only quantities the storage correction depends on. Both near
    zero means the vessels dominate and Brace applies; of order one means the
    plug stores as much gas as a vessel does and ignoring it biases k low.
    """
    _require_positive("upstream_volume_cm3", upstream_volume_cm3)
    _require_positive("downstream_volume_cm3", downstream_volume_cm3)
    if pore_volume_cm3 < 0.0 or not math.isfinite(pore_volume_cm3):
        raise PulseDecayInputError(
            f"pore_volume_cm3 must be finite and non-negative, got {pore_volume_cm3!r}."
        )
    return (
        pore_volume_cm3 / upstream_volume_cm3,
        pore_volume_cm3 / downstream_volume_cm3,
    )


def _storage_residual(theta: float, a1: float, a2: float) -> float:
    """The pole-free form of the Dicker-Smits eigenvalue equation.

    The textbook statement is ``tan(theta) = theta(a1+a2)/(theta^2 - a1 a2)``,
    which has a **pole** at ``theta = sqrt(a1 a2)``. Bracketing a root finder on
    ``(0, pi/2)`` -- the interval every derivation quotes -- divides by zero
    there, and once ``a1 a2 > (pi/2)^2`` the bracket is inverted outright. Both
    failures are reachable with vessels small enough to give a workable run
    time.

    Multiplying through by ``cos(theta)(theta^2 - a1 a2)`` removes the pole and
    leaves a smooth function with exactly one sign change on ``(0, pi)`` for any
    positive ratios -- verified by dense scan over ``a in [1e-6, 10]^2``.
    """
    return (theta * theta - a1 * a2) * math.sin(theta) - theta * (a1 + a2) * math.cos(
        theta
    )


def first_storage_root(upstream_ratio: float, downstream_ratio: float) -> float:
    """``theta_1``: the slowest decay mode of a plug with its own storage.

    Args:
        upstream_ratio: ``a1 = V_pore / V1``.
        downstream_ratio: ``a2 = V_pore / V2``.

    Returns:
        The root in ``(0, pi)``. It tends to ``sqrt(a1 + a2)`` as the ratios go
        to zero (recovering Brace) and to ``pi`` as they grow without bound
        (both ends effectively no-flux).

    Raises:
        PulseDecayInputError: either ratio is negative or non-finite, or both
            are negligible -- there is no root at ``a1 = a2 = 0`` because the
            model has degenerated to the zero-storage form, which the caller
            should use directly.
    """
    for name, value in (
        ("upstream_ratio", upstream_ratio),
        ("downstream_ratio", downstream_ratio),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise PulseDecayInputError(
                f"{name} must be finite and non-negative, got {value!r}."
            )
    if upstream_ratio + downstream_ratio <= _NEGLIGIBLE_STORAGE:
        raise PulseDecayInputError(
            "Both storage ratios are negligible, so the Dicker-Smits equation has "
            "no root: the model has reduced to the zero-storage (Brace) form. Call "
            "brace_permeability_darcy directly."
        )

    from scipy.optimize import brentq

    # g(0+) < 0 and g(pi) = +pi(a1+a2) > 0 for any positive ratios, so the
    # bracket is guaranteed to contain the sign change.
    lower, upper = _ROOT_TOLERANCE, math.pi - _ROOT_TOLERANCE
    return float(
        brentq(
            _storage_residual,
            lower,
            upper,
            args=(upstream_ratio, downstream_ratio),
            xtol=_ROOT_TOLERANCE,
        )
    )


# --------------------------------------------------------------------------
# Permeability from a decay rate
# --------------------------------------------------------------------------


def brace_permeability_darcy(
    *,
    decay_rate_per_s: float,
    viscosity_cp: float,
    gas_compressibility_per_atm: float,
    length_cm: float,
    area_cm2: float,
    upstream_volume_cm3: float,
    downstream_volume_cm3: float,
) -> float:
    """Zero-storage permeability, darcy.

    ``k = alpha * mu * c_g * L / (A * (1/V1 + 1/V2))``.

    Valid when the plug's pore volume is small against both vessels. Otherwise
    it reads **low**, because part of the observed decay went into filling the
    plug rather than through it -- use :func:`dicker_smits_permeability_darcy`.
    """
    _require_positive("decay_rate_per_s", decay_rate_per_s)
    _require_positive("viscosity_cp", viscosity_cp)
    _require_positive("gas_compressibility_per_atm", gas_compressibility_per_atm)
    _require_positive("length_cm", length_cm)
    _require_positive("area_cm2", area_cm2)
    _require_positive("upstream_volume_cm3", upstream_volume_cm3)
    _require_positive("downstream_volume_cm3", downstream_volume_cm3)

    inverse_volume = 1.0 / upstream_volume_cm3 + 1.0 / downstream_volume_cm3
    return (
        decay_rate_per_s
        * viscosity_cp
        * gas_compressibility_per_atm
        * length_cm
        / (area_cm2 * inverse_volume)
    )


def dicker_smits_permeability_darcy(
    *,
    decay_rate_per_s: float,
    viscosity_cp: float,
    gas_compressibility_per_atm: float,
    length_cm: float,
    area_cm2: float,
    porosity_fraction: float,
    upstream_volume_cm3: float,
    downstream_volume_cm3: float,
) -> float:
    """Storage-corrected permeability, darcy.

    ``k = alpha * phi * mu * c_g * L^2 / theta_1^2``.

    ``theta_1`` depends only on the storage ratios, not on ``k``, so this is
    explicit -- no iteration. Always at least as large as the Brace result, and
    the gap is the fraction of the decay that filled the plug instead of
    crossing it.
    """
    _require_positive("decay_rate_per_s", decay_rate_per_s)
    _require_positive("viscosity_cp", viscosity_cp)
    _require_positive("gas_compressibility_per_atm", gas_compressibility_per_atm)
    _require_positive("length_cm", length_cm)
    _require_positive("area_cm2", area_cm2)
    _require_positive("porosity_fraction", porosity_fraction)

    pore = pore_volume_cm3(
        area_cm2=area_cm2, length_cm=length_cm, porosity_fraction=porosity_fraction
    )
    a1, a2 = storage_ratios(
        pore_volume_cm3=pore,
        upstream_volume_cm3=upstream_volume_cm3,
        downstream_volume_cm3=downstream_volume_cm3,
    )
    if a1 + a2 <= _NEGLIGIBLE_STORAGE:
        # Degenerate storage: the correction has nothing to correct.
        return brace_permeability_darcy(
            decay_rate_per_s=decay_rate_per_s,
            viscosity_cp=viscosity_cp,
            gas_compressibility_per_atm=gas_compressibility_per_atm,
            length_cm=length_cm,
            area_cm2=area_cm2,
            upstream_volume_cm3=upstream_volume_cm3,
            downstream_volume_cm3=downstream_volume_cm3,
        )
    theta = first_storage_root(a1, a2)
    return (
        decay_rate_per_s
        * porosity_fraction
        * viscosity_cp
        * gas_compressibility_per_atm
        * length_cm**2
        / theta**2
    )


def brace_decay_rate_per_s(
    *,
    permeability_darcy: float,
    viscosity_cp: float,
    gas_compressibility_per_atm: float,
    length_cm: float,
    area_cm2: float,
    upstream_volume_cm3: float,
    downstream_volume_cm3: float,
) -> float:
    """Inverse of :func:`brace_permeability_darcy`, for predicting a run."""
    _require_positive("permeability_darcy", permeability_darcy)
    _require_positive("viscosity_cp", viscosity_cp)
    _require_positive("gas_compressibility_per_atm", gas_compressibility_per_atm)
    _require_positive("length_cm", length_cm)
    _require_positive("area_cm2", area_cm2)
    _require_positive("upstream_volume_cm3", upstream_volume_cm3)
    _require_positive("downstream_volume_cm3", downstream_volume_cm3)

    inverse_volume = 1.0 / upstream_volume_cm3 + 1.0 / downstream_volume_cm3
    return (
        permeability_darcy
        * area_cm2
        * inverse_volume
        / (viscosity_cp * gas_compressibility_per_atm * length_cm)
    )


def dicker_smits_decay_rate_per_s(
    *,
    permeability_darcy: float,
    viscosity_cp: float,
    gas_compressibility_per_atm: float,
    length_cm: float,
    area_cm2: float,
    porosity_fraction: float,
    upstream_volume_cm3: float,
    downstream_volume_cm3: float,
) -> float:
    """Inverse of :func:`dicker_smits_permeability_darcy`."""
    _require_positive("permeability_darcy", permeability_darcy)
    _require_positive("porosity_fraction", porosity_fraction)

    pore = pore_volume_cm3(
        area_cm2=area_cm2, length_cm=length_cm, porosity_fraction=porosity_fraction
    )
    a1, a2 = storage_ratios(
        pore_volume_cm3=pore,
        upstream_volume_cm3=upstream_volume_cm3,
        downstream_volume_cm3=downstream_volume_cm3,
    )
    if a1 + a2 <= _NEGLIGIBLE_STORAGE:
        return brace_decay_rate_per_s(
            permeability_darcy=permeability_darcy,
            viscosity_cp=viscosity_cp,
            gas_compressibility_per_atm=gas_compressibility_per_atm,
            length_cm=length_cm,
            area_cm2=area_cm2,
            upstream_volume_cm3=upstream_volume_cm3,
            downstream_volume_cm3=downstream_volume_cm3,
        )
    theta = first_storage_root(a1, a2)
    return (
        permeability_darcy
        * theta**2
        / (
            porosity_fraction
            * viscosity_cp
            * gas_compressibility_per_atm
            * length_cm**2
        )
    )


# --------------------------------------------------------------------------
# Finding the pulse and the part of the decay worth fitting
# --------------------------------------------------------------------------


def find_pulse(
    times_s: Sequence[float],
    delta_pressure_atm: Sequence[float],
    *,
    median_window: int = 5,
) -> tuple[int, float]:
    """Locate the applied pulse: ``(index, dP0)`` at its peak.

    The peak is *located* by the argmax of a short **moving median** rather than
    the raw argmax -- the valve opening is monotone-up over seconds while the
    decay runs for minutes to hours, so the peak is unambiguous, but a single
    noise spike is not a pulse and a median ignores it.

    The amplitude reported is then the **raw** sample at that index, not the
    smoothed one. A centred median is biased low at a corner, and the pulse peak
    is exactly a corner: rise, then decay. Smoothing it would understate dP0 by
    several percent and shift every fit-window boundary with it. A spike cannot
    slip through, because a lone outlier does not move the median's argmax.

    Raises:
        PulseDecayInputError: the two series differ in length or are empty.
    """
    if len(times_s) != len(delta_pressure_atm):
        raise PulseDecayInputError(
            f"times ({len(times_s)}) and delta pressures "
            f"({len(delta_pressure_atm)}) must be the same length."
        )
    if not times_s:
        raise PulseDecayInputError("no samples to search for a pulse.")

    values = list(delta_pressure_atm)
    half = max(0, median_window // 2)
    smoothed = [
        statistics.median(values[max(0, i - half) : i + half + 1])
        for i in range(len(values))
    ]
    peak = max(range(len(smoothed)), key=smoothed.__getitem__)
    return peak, values[peak]


def fit_window(
    times_s: Sequence[float],
    delta_pressure_atm: Sequence[float],
    *,
    peak_index: int,
    peak_value: float,
    start_fraction: float,
    end_fraction: float,
) -> tuple[int, int]:
    """The slice of the decay to fit: ``(start, end)`` indices, end exclusive.

    Runs from the first sample at or below ``start_fraction * dP0`` to the first
    at or below ``end_fraction * dP0``.

    Skipping the very top of the decay is not cosmetic, and there are two
    independent reasons for it:

    1. the valve-opening transient, which is not a decay at all;
    2. the single exponential is **asymptotic, not exact** -- with finite sample
       storage the solution is an infinite series whose higher modes decay much
       faster, so fitting from the peak biases alpha *high*.

    Stopping before the noise floor is the mirror image: late samples carry
    almost no signal, and including them adds scatter and bias rather than
    information.
    """
    if not 0.0 < end_fraction < start_fraction <= 1.0:
        raise PulseDecayInputError(
            f"need 0 < end_fraction ({end_fraction}) < start_fraction "
            f"({start_fraction}) <= 1."
        )
    if peak_value <= 0.0:
        raise PulseDecayInputError(
            f"the pulse amplitude must be positive, got {peak_value!r}."
        )

    start_level = start_fraction * peak_value
    end_level = end_fraction * peak_value

    start = peak_index
    while start < len(delta_pressure_atm) and delta_pressure_atm[start] > start_level:
        start += 1

    end = start
    while end < len(delta_pressure_atm) and delta_pressure_atm[end] > end_level:
        end += 1

    return start, min(end + 1, len(delta_pressure_atm))


def _bin_series(
    times_s: Sequence[float], values: Sequence[float], bin_s: float
) -> tuple[list[float], list[float]]:
    """Average into fixed-width time bins, keeping only non-empty ones."""
    if not times_s:
        return [], []
    start = times_s[0]
    bins: dict[int, list[tuple[float, float]]] = {}
    for t, v in zip(times_s, values):
        bins.setdefault(int((t - start) // bin_s), []).append((t, v))
    binned_t: list[float] = []
    binned_v: list[float] = []
    for key in sorted(bins):
        rows = bins[key]
        binned_t.append(sum(t for t, _ in rows) / len(rows))
        binned_v.append(sum(v for _, v in rows) / len(rows))
    return binned_t, binned_v


def _log_linear_rate(
    times_s: Sequence[float], values: Sequence[float]
) -> tuple[float, float]:
    """Closed-form OLS of ``ln(dP)`` against t -> ``(alpha, intercept)``.

    Only the positive samples participate, since the log of a noise-dominated
    negative reading is undefined. Used as the seed for the nonlinear fit and as
    the monitor's running estimate, where a scipy call per sample would be
    absurd -- the same reasoning ``steady_state._ols_slope`` already documents.
    """
    pairs = [(t, v) for t, v in zip(times_s, values) if v > 0.0]
    if len(pairs) < 2:
        raise PulseDecayInputError(
            "fewer than two positive differential pressures, so no decay can be fitted."
        )
    n = len(pairs)
    mean_t = sum(t for t, _ in pairs) / n
    logs = [math.log(v) for _, v in pairs]
    mean_y = sum(logs) / n
    s_tt = sum((t - mean_t) ** 2 for t, _ in pairs)
    if s_tt <= 0.0:
        raise PulseDecayInputError(
            "every sample shares one timestamp, so no decay rate can be fitted."
        )
    s_ty = sum((t - mean_t) * (y - mean_y) for (t, _), y in zip(pairs, logs))
    slope = s_ty / s_tt
    return -slope, mean_y - slope * mean_t


def _lag_one_autocorrelation(residuals: Sequence[float]) -> float | None:
    """Lag-1 autocorrelation, the cheapest test for structure in the residuals."""
    if len(residuals) < 3:
        return None
    mean = sum(residuals) / len(residuals)
    centred = [r - mean for r in residuals]
    denominator = sum(c * c for c in centred)
    if denominator <= 0.0:
        return None
    numerator = sum(a * b for a, b in zip(centred, centred[1:]))
    return numerator / denominator


def fit_decay_rate(
    times_s: Sequence[float],
    delta_pressure_atm: Sequence[float],
    *,
    fit_offset: bool = True,
    bin_s: float | None = 1.0,
) -> DecayFit:
    """Fit ``dP = A exp(-alpha (t - t0)) + C`` and report alpha with its uncertainty.

    **Why nonlinear rather than log-linear.** The two transducers reading P1 and
    P2 are independent instruments with independent zero errors, so their
    difference has an offset that no amount of averaging removes. ``log(A e^-at
    + C)`` flattens at late time, so a log-linear fit returns a systematically
    *low* alpha and hence a low permeability -- measurably so: a 0.5 kPa offset
    on a 50 kPa pulse biases alpha by -5%, and 5 kPa biases it by -33%. Fitting
    the offset as a free parameter removes that entirely.

    **Why binning.** At 10 Hz over hours the residuals are strongly
    autocorrelated -- thermal drift and transducer 1/f, not white noise -- so
    ``u(alpha)`` computed over half a million nominally independent samples is
    optimistic by roughly ``sqrt(n/n_eff)``, which would drive the effective
    degrees of freedom absurdly high and report a confidently wrong interval.
    Binning first makes the remaining samples much closer to independent; the
    reported lag-1 autocorrelation says whether it worked.

    Args:
        times_s: Elapsed seconds, ascending.
        delta_pressure_atm: P1 - P2 at those times, atm.
        fit_offset: Fit the constant term. Leave on unless you know the two
            transducers share a zero.
        bin_s: Bin width before fitting. ``None`` fits every sample.

    Returns:
        The fit, including the diagnostics needed to judge it.

    Raises:
        PulseDecayInputError: not enough usable samples, or the series is flat.
    """
    if len(times_s) != len(delta_pressure_atm):
        raise PulseDecayInputError(
            f"times ({len(times_s)}) and delta pressures "
            f"({len(delta_pressure_atm)}) must be the same length."
        )
    raw_count = len(times_s)
    if raw_count < 3:
        raise PulseDecayInputError(
            f"need at least 3 samples to fit a decay, got {raw_count}."
        )

    if bin_s is not None and bin_s > 0.0:
        fit_times, fit_values = _bin_series(times_s, delta_pressure_atm, bin_s)
    else:
        fit_times, fit_values = list(times_s), list(delta_pressure_atm)
    if len(fit_times) < 3:
        raise PulseDecayInputError(
            f"binning at {bin_s} s left only {len(fit_times)} points; use a shorter "
            "bin or record a longer decay."
        )

    start_s, end_s = fit_times[0], fit_times[-1]
    # Fit against time since the window start rather than since the run start:
    # with alpha ~ 1e-5 and t ~ 1e4, the product is well conditioned, while an
    # offset origin would make the amplitude an extrapolation over many decades.
    shifted = [t - start_s for t in fit_times]

    seed_alpha, seed_intercept = _log_linear_rate(shifted, fit_values)
    seed_amplitude = math.exp(seed_intercept)

    model = "log_linear"
    alpha = seed_alpha
    amplitude = seed_amplitude
    offset: float | None = None
    u_alpha: float | None = None
    amplitude_offset_correlation: float | None = None
    parameters = 2

    if fit_offset:
        try:
            import warnings as _warnings

            import numpy as np
            from scipy.optimize import OptimizeWarning, curve_fit

            def _model(t, a, rate, c):
                return a * np.exp(-rate * t) + c

            seed_offset = min(fit_values)
            with _warnings.catch_warnings():
                # "Covariance of the parameters could not be estimated" is
                # expected on a clean decay and is handled below; it is not
                # something to print at an operator mid-run.
                _warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(
                    _model,
                    np.asarray(shifted, dtype=float),
                    np.asarray(fit_values, dtype=float),
                    p0=[max(seed_amplitude, 1e-12), max(seed_alpha, 1e-12), seed_offset],
                    maxfev=20_000,
                )
            if np.all(np.isfinite(popt)) and popt[1] > 0.0:
                amplitude, alpha, offset = (float(popt[0]), float(popt[1]), float(popt[2]))
                model = "exponential_offset"
                parameters = 3
                # A noiseless (or near-noiseless) decay leaves curve_fit unable
                # to estimate a covariance, which is not a reason to throw away
                # a good fit -- only a reason to report no uncertainty from it.
                # The log-linear fallback below then supplies one.
                if np.all(np.isfinite(pcov)):
                    u_alpha = float(math.sqrt(abs(pcov[1][1])))
                    spread = math.sqrt(abs(pcov[0][0]) * abs(pcov[2][2]))
                    if spread > 0.0:
                        amplitude_offset_correlation = float(pcov[0][2] / spread)
        except Exception:  # noqa: BLE001 - any solver failure falls back
            # A non-convergent solver is not a reason to lose the run; the
            # log-linear result is recorded, and `model` says which was used.
            model = "log_linear"

    def _predict(t: float) -> float:
        return amplitude * math.exp(-alpha * t) + (offset or 0.0)

    residuals = [v - _predict(t) for t, v in zip(shifted, fit_values)]
    mean_value = sum(fit_values) / len(fit_values)
    ss_total = sum((v - mean_value) ** 2 for v in fit_values)
    ss_residual = sum(r * r for r in residuals)
    r_squared = 1.0 if ss_total <= 0.0 else 1.0 - ss_residual / ss_total

    if u_alpha is None and len(fit_times) > parameters:
        # Log-linear fallback: propagate the slope's standard error through the
        # log, which is exact for the slope of ln(dP) against t.
        positive = [(t, v) for t, v in zip(shifted, fit_values) if v > 0.0]
        if len(positive) > 2:
            mean_t = sum(t for t, _ in positive) / len(positive)
            s_tt = sum((t - mean_t) ** 2 for t, _ in positive)
            logs = [math.log(v) for _, v in positive]
            predicted = [
                math.log(max(_predict(t), 1e-300)) for t, _ in positive
            ]
            log_residual = sum(
                (y - p) ** 2 for y, p in zip(logs, predicted)
            ) / (len(positive) - 2)
            if s_tt > 0.0 and log_residual > 0.0:
                u_alpha = math.sqrt(log_residual / s_tt)

    return DecayFit(
        decay_rate_per_s=alpha,
        decay_rate_standard_uncertainty_per_s=u_alpha,
        degrees_of_freedom=max(float(len(fit_times) - parameters), 1.0),
        amplitude_atm=amplitude,
        offset_atm=offset,
        r_squared=r_squared,
        start_elapsed_s=start_s,
        end_elapsed_s=end_s,
        sample_count=len(fit_times),
        raw_sample_count=raw_count,
        model=model,
        residual_autocorrelation=_lag_one_autocorrelation(residuals),
        amplitude_offset_correlation=amplitude_offset_correlation,
    )


# --------------------------------------------------------------------------
# Live monitoring
# --------------------------------------------------------------------------


class PulseDecayMonitor:
    """Streaming view of a decay: where it is, and when it will finish.

    The live analogue of :class:`~gasperm.steady_state.SteadyStateDetector`. It
    exists because a pulse-decay run takes hours and the operator needs to see
    progress; it also supplies the running permeability the console and the live
    plot show. The definitive fit happens once, at the end, in
    :func:`fit_decay_rate` -- this only ever runs a closed-form log-linear OLS,
    so it costs nothing per sample.
    """

    def __init__(self, config: PulseDecayConfig, *, min_pulse_atm: float) -> None:
        self.config = config
        self.min_pulse_atm = min_pulse_atm
        self._peak_value: float = 0.0
        self._peak_time: float | None = None
        self._min_since_peak: float | None = None
        self._reversed = False
        self._status = PulseDecayStatus()
        self._reset_accumulator()

    def _reset_accumulator(self) -> None:
        """Start the running regression over.

        Called whenever a new peak is found, because the fit window's bounds are
        fractions of the peak: once dP0 changes, everything accumulated against
        the old bounds is against the wrong window.
        """
        self._n = 0
        self._t0: float | None = None
        self._sum_t = 0.0
        self._sum_y = 0.0
        self._sum_tt = 0.0
        self._sum_ty = 0.0

    @property
    def status(self) -> PulseDecayStatus:
        """The most recent verdict."""
        return self._status

    @property
    def is_complete(self) -> bool:
        """Whether the decay has fallen past ``stop_below_fraction``."""
        return self._status.phase == "complete"

    @property
    def pulse_amplitude_atm(self) -> float | None:
        return self._peak_value if self._peak_time is not None else None

    def update(self, elapsed_s: float, delta_pressure_atm: float) -> PulseDecayStatus:
        """Add one sample and re-evaluate.

        **O(1) in both time and memory.** A pulse-decay run is hours long -- half
        a million samples at 10 Hz -- so the running regression is kept as five
        accumulated sums rather than recomputed from a stored series. Anything
        per-sample that scanned the history would make the acquisition loop
        quadratic and eventually miss its sample slots.

        Args:
            elapsed_s: Seconds since the run started.
            delta_pressure_atm: P1 - P2 for this sample.

        Returns:
            The current status, safe to display every sample.
        """
        if delta_pressure_atm >= self.min_pulse_atm and (
            self._peak_time is None or delta_pressure_atm > self._peak_value
        ):
            # Still rising, or rising again. Either way this is the new peak;
            # a *second* rise, after the decay was properly under way, is a leak
            # or a reopened valve and is recorded as such.
            if self._peak_time is not None and self._is_past_transient():
                self._reversed = True
            self._peak_value = delta_pressure_atm
            self._peak_time = elapsed_s
            self._min_since_peak = delta_pressure_atm
            self._reset_accumulator()
        elif self._peak_time is not None:
            self._min_since_peak = (
                delta_pressure_atm
                if self._min_since_peak is None
                else min(self._min_since_peak, delta_pressure_atm)
            )
            self._accumulate(elapsed_s, delta_pressure_atm)

        self._status = self._evaluate(elapsed_s, delta_pressure_atm)
        return self._status

    def _accumulate(self, elapsed_s: float, delta_pressure_atm: float) -> None:
        """Fold one sample into the running log-linear regression, if it counts."""
        if self._peak_value <= 0.0 or delta_pressure_atm <= 0.0:
            return
        upper = self.config.fit_start_fraction * self._peak_value
        lower = self.config.fit_end_fraction * self._peak_value
        if not lower <= delta_pressure_atm <= upper:
            return
        if self._t0 is None:
            self._t0 = elapsed_s
        t = elapsed_s - self._t0
        y = math.log(delta_pressure_atm)
        self._n += 1
        self._sum_t += t
        self._sum_y += y
        self._sum_tt += t * t
        self._sum_ty += t * y

    def _running_rate(self) -> float | None:
        """The accumulated regression's slope, as a positive decay rate."""
        if self._n < 3:
            return None
        denominator = self._n * self._sum_tt - self._sum_t * self._sum_t
        if denominator <= 0.0:
            return None
        slope = (self._n * self._sum_ty - self._sum_t * self._sum_y) / denominator
        return -slope if slope < 0.0 else None

    def _is_past_transient(self) -> bool:
        """Whether the decay has moved far enough down to call the peak settled."""
        if self._peak_value <= 0.0 or self._min_since_peak is None:
            return False
        return self._min_since_peak < 0.9 * self._peak_value

    def _evaluate(self, elapsed_s: float, delta_pressure_atm: float) -> PulseDecayStatus:
        if self._peak_time is None:
            return PulseDecayStatus(
                phase="waiting",
                elapsed_s=elapsed_s,
                delta_pressure_atm=delta_pressure_atm,
                summary=(
                    f"waiting for a pulse of at least {self.min_pulse_atm:.4g} atm"
                ),
            )

        fraction = (
            delta_pressure_atm / self._peak_value if self._peak_value > 0.0 else None
        )
        rate = self._running_rate()

        complete = fraction is not None and fraction <= self.config.stop_below_fraction
        if complete:
            phase = "complete"
        elif self._is_past_transient():
            phase = "decaying"
        else:
            phase = "transient"

        projected: float | None = None
        if rate and fraction and not complete and self._peak_time is not None:
            remaining = math.log(fraction / self.config.stop_below_fraction) / rate
            projected = elapsed_s + remaining

        if phase == "complete":
            summary = f"decay complete at dP/dP0 = {fraction:.3f}"
        elif phase == "transient":
            summary = "pulse applied, still settling"
        elif rate:
            summary = f"decaying, tau = {1.0 / rate:.0f} s"
        else:
            summary = "decaying, collecting the fit window"

        return PulseDecayStatus(
            phase=phase,
            elapsed_s=elapsed_s,
            delta_pressure_atm=delta_pressure_atm,
            pulse_at_elapsed_s=self._peak_time,
            pulse_amplitude_atm=self._peak_value,
            decay_fraction=fraction,
            decay_rate_per_s=rate,
            time_constant_s=(1.0 / rate) if rate else None,
            projected_complete_elapsed_s=projected,
            fit_sample_count=self._n,
            reversed_since_peak=self._reversed,
            summary=summary,
        )
