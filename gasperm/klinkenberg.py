"""Klinkenberg slippage correction.

Apparent gas permeability depends on mean pore pressure because gas molecules
slip at the pore wall. Extrapolating to infinite pressure (``1/P -> 0``)
recovers the liquid-equivalent permeability::

    k_g = k_L + k_L * (b / P_mean)

which is linear in ``1 / P_mean``::

    k_g = k_L + (k_L * b) * (1 / P_mean)
           ^          ^
           intercept  slope

Hardware-free and independently testable against synthetic ``(P_mean, k_g)``
pairs generated from a known ``k_L``/``b``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

from gasperm import units
from gasperm.models import KlinkenbergPoint, KlinkenbergResult

__all__ = [
    "fit_klinkenberg",
    "load_points_from_csv",
    "MIN_POINTS",
    "RECOMMENDED_POINTS",
    "POOR_FIT_R_SQUARED",
]

#: Two points define a line exactly but leave nothing to cross-check.
MIN_POINTS = 2
RECOMMENDED_POINTS = 3
#: Below this R^2 the points are not convincingly linear in 1/P.
POOR_FIT_R_SQUARED = 0.95


def fit_klinkenberg(points: Sequence[KlinkenbergPoint]) -> KlinkenbergResult:
    """Regress ``k_g`` against ``1 / P_mean`` and recover ``k_L`` and ``b``.

    Ordinary least squares is sufficient here: the points are few, the
    relationship is genuinely linear over the usable pressure range, and the
    scatter is dominated by run-to-run steady-state error rather than by
    anything structured.

    Args:
        points: Two or more ``(mean pressure, apparent permeability)`` pairs
            for the **same** sample, at different mean pressures.

    Returns:
        The fit, including R^2 and any warnings worth showing the operator.

    Raises:
        ValueError: fewer than two points, duplicate mean pressures (no spread
            to regress over), or non-finite values.
    """
    if len(points) < MIN_POINTS:
        raise ValueError(
            f"Klinkenberg regression needs at least {MIN_POINTS} points at different "
            f"mean pressures, got {len(points)}."
        )

    sample_ids = {p.sample_id for p in points if p.sample_id}
    inverse_pressure = np.array([p.inverse_mean_pressure for p in points], dtype=float)
    apparent_k = np.array([p.apparent_permeability_darcy for p in points], dtype=float)

    if not np.all(np.isfinite(inverse_pressure)) or not np.all(np.isfinite(apparent_k)):
        raise ValueError(
            "Klinkenberg input contains non-finite values; check that every run "
            "produced a usable steady-state permeability."
        )
    if np.ptp(inverse_pressure) == 0.0:
        raise ValueError(
            "All points share the same mean pressure, so there is no pressure spread "
            "to extrapolate over. Run the sample at genuinely different mean pressures."
        )

    regression = stats.linregress(inverse_pressure, apparent_k)
    slope = float(regression.slope)
    intercept = float(regression.intercept)
    r_squared = float(regression.rvalue**2)

    warnings: list[str] = []

    if len(points) < RECOMMENDED_POINTS:
        warnings.append(
            f"Only {len(points)} points supplied. Two points fit an exact line, so R^2 "
            "is meaningless and there is no way to detect a bad run. Use at least "
            f"{RECOMMENDED_POINTS}."
        )
    elif r_squared < POOR_FIT_R_SQUARED:
        warnings.append(
            f"Poor linear fit (R^2 = {r_squared:.4f} < {POOR_FIT_R_SQUARED}). The points "
            "are not behaving linearly in 1/P_mean -- check for non-steady-state runs, "
            "a leaking sleeve, or turbulent (non-Darcy) flow at the highest rate."
        )

    if intercept <= 0.0:
        warnings.append(
            f"The fitted intercept (k_L = {intercept:.6g} D) is not positive, which is "
            "non-physical. The pressure range is probably too narrow, or one run is an "
            "outlier."
        )
    if slope < 0.0:
        warnings.append(
            "The fitted slope is negative, implying a negative slippage factor. "
            "Klinkenberg predicts apparent permeability to *fall* as mean pressure "
            "rises; check whether the mean pressures are paired with the right runs."
        )
    if len(sample_ids) > 1:
        warnings.append(
            "Points come from more than one sample id ("
            + ", ".join(sorted(sample_ids))
            + "). The Klinkenberg correction is only valid within a single sample."
        )

    # b = slope / intercept. Guard the division rather than emitting inf.
    if intercept != 0.0:
        slippage_factor = slope / intercept
    else:
        slippage_factor = math.nan
        warnings.append(
            "The intercept is exactly zero, so the slippage factor b = slope / k_L is "
            "undefined."
        )

    return KlinkenbergResult(
        liquid_permeability_darcy=intercept,
        slippage_factor_atm=slippage_factor,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        intercept_stderr=float(regression.intercept_stderr),
        slope_stderr=float(regression.stderr),
        points=list(points),
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# CSV input
# --------------------------------------------------------------------------

_PRESSURE_COLUMN_CANDIDATES = (
    "mean_pressure",
    "p_mean",
    "pmean",
    "mean_pore_pressure",
    "mean_pressure_atm",
)
_PERMEABILITY_COLUMN_CANDIDATES = (
    "apparent_permeability",
    "k_g",
    "kg",
    "permeability",
    "apparent_permeability_darcy",
)


def _find_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    """First column whose normalised name starts with one of ``candidates``."""
    normalised = {name.strip().lower(): name for name in fieldnames}
    # Exact match first, so "mean_pressure_atm" beats a prefix hit elsewhere.
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    for candidate in candidates:
        for lowered, original in normalised.items():
            if lowered.startswith(candidate):
                return original
    return None


def _unit_suffix(column: str, known: Sequence[str], default: str) -> str:
    """Read a trailing unit off a column name, e.g. ``k_g_mD`` -> ``mD``."""
    tail = column.strip().lower().rsplit("_", 1)[-1]
    for unit in known:
        if tail == unit.lower():
            return unit
    return default


def load_points_from_csv(
    path: str | Path,
    *,
    pressure_unit: str | None = None,
    permeability_unit: str | None = None,
) -> list[KlinkenbergPoint]:
    """Read ``(mean pressure, apparent permeability)`` pairs from a CSV.

    Column names are matched case-insensitively against the usual spellings
    (``mean_pressure``/``p_mean``, ``apparent_permeability``/``k_g``). Units
    come from the explicit arguments if given, otherwise from a trailing unit
    in the column name (``mean_pressure_kPa``, ``k_g_mD``), otherwise
    defaulting to atm and mD respectively.

    Args:
        path: CSV file.
        pressure_unit: Overrides any unit inferred from the header.
        permeability_unit: Overrides any unit inferred from the header.

    Returns:
        Points converted into internal CGS units (atm, darcy).

    Raises:
        ValueError: the file is empty, lacks a recognisable column pair, or
            contains an unparseable row.
    """
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row.")

        pressure_column = _find_column(reader.fieldnames, _PRESSURE_COLUMN_CANDIDATES)
        permeability_column = _find_column(
            reader.fieldnames, _PERMEABILITY_COLUMN_CANDIDATES
        )
        if pressure_column is None or permeability_column is None:
            raise ValueError(
                f"{csv_path} needs a mean-pressure column (one of "
                f"{', '.join(_PRESSURE_COLUMN_CANDIDATES)}) and an apparent-permeability "
                f"column (one of {', '.join(_PERMEABILITY_COLUMN_CANDIDATES)}). "
                f"Found: {', '.join(reader.fieldnames)}"
            )

        resolved_pressure_unit = pressure_unit or _unit_suffix(
            pressure_column, sorted(units.SUPPORTED_PRESSURE_UNITS), "atm"
        )
        resolved_permeability_unit = permeability_unit or _unit_suffix(
            permeability_column, sorted(units.SUPPORTED_PERMEABILITY_UNITS), "mD"
        )
        label_column = _find_column(reader.fieldnames, ("label", "run", "name"))
        sample_column = _find_column(reader.fieldnames, ("sample_id", "sample"))

        points: list[KlinkenbergPoint] = []
        for row_number, row in enumerate(reader, start=2):
            raw_pressure = (row.get(pressure_column) or "").strip()
            raw_permeability = (row.get(permeability_column) or "").strip()
            if not raw_pressure and not raw_permeability:
                continue  # tolerate blank separator rows
            try:
                pressure_value = float(raw_pressure)
                permeability_value = float(raw_permeability)
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path} line {row_number}: could not read a number from "
                    f"{pressure_column}={raw_pressure!r}, "
                    f"{permeability_column}={raw_permeability!r}"
                ) from exc

            pressure_atm = units.to_atm(pressure_value, resolved_pressure_unit)
            if pressure_atm <= 0.0:
                raise ValueError(
                    f"{csv_path} line {row_number}: mean pressure must be positive and "
                    f"absolute, got {pressure_value} {resolved_pressure_unit}."
                )
            points.append(
                KlinkenbergPoint(
                    mean_pressure_atm=pressure_atm,
                    apparent_permeability_darcy=units.darcy_from(
                        permeability_value, resolved_permeability_unit
                    ),
                    label=(row.get(label_column) or "").strip() if label_column else "",
                    sample_id=(row.get(sample_column) or "").strip() or None
                    if sample_column
                    else None,
                    source_path=str(csv_path),
                )
            )

    if not points:
        raise ValueError(f"{csv_path} contained no data rows.")
    return points
