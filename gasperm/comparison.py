"""Comparing two measurements of permeability, per the GUM's treatment of a difference.

Hardware-free and config-free: everything here takes stored results and returns
plain models, so the statistics are testable against hand-computed numbers with
no rig, no CSV and no CoolProp.

**The measurand is the change, not either value.** That single fact reorganises
the whole uncertainty calculation, and it is what makes a before/after study
far more sensitive than the absolute numbers suggest.

For a ratio ``R = k_B / k_A`` of two quantities built from the same input
components, GUM 5.2 gives

    u_rel^2(R) = SUM_i [ c_i,A u_i,A - c_i,B u_i,B ]^2      over SHARED inputs
               + SUM_j [ (c_j,A u_j,A)^2 + (c_j,B u_j,B)^2 ]  over INDEPENDENT inputs

An input is **shared** when the same physical error produced both readings: the
same plug measured with the same calipers, the same transducer on the same
calibration, the same flowmeter, the same viscosity model. Its error moves both
measurements the same way and is *absent* from their ratio.

Note what the shared term does when the two readings differ: it is a difference
of contributions, not zero. Two runs at 10.0 and 10.2 atm on a percent-of-full-
scale transducer share an *absolute* error, so their relative contributions
differ slightly and the formula charges exactly that residue -- automatically,
with no special case. Matched conditions are therefore not a precondition this
module asserts; they are a quantity it prices.

What never cancels is Type A: the fit scatter, the flow noise, the repeatability.
Those set the detection limit, and on a well-matched pair they are essentially
all that is left. A rig reporting U(k) = 20% on each of two runs can still
resolve a 5% change between them.

Three consequences worth stating, because they drive the code below:

* **Geometry cancels only if it was not re-measured.** ``L`` and ``d`` enter
  ``k`` as ``L/d^2``, so a 1% length error scales every ``k`` on that plug by
  1% and vanishes from the ratio -- but only while it is literally the same
  number. If the plug was re-measured between campaigns, the two errors are
  independent draws and the cancellation is void. This module detects that from
  the values themselves rather than trusting an assertion.
* **Rig-level inputs cancel even between different plugs.** Two plugs measured
  on the same rig share the transducer, meter and gas model, so those cancel
  from *their* ratio too. Sharing is therefore decided per component, never by
  one paired/unpaired switch.
* **k_L carries a systematic its own fit cannot see.** Scaling every ``k_g`` in
  a series by ``(1+e)`` scales the intercept by ``(1+e)`` and leaves the
  slippage factor ``b`` untouched. A regression's intercept standard error
  describes only the scatter *about* the line, so the honest
  ``u(k_L)`` is that in quadrature with the series' common systematic -- and in
  a paired comparison the systematic half is exactly the half that cancels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from gasperm.models import (
    ComparisonResult,
    ComponentPairing,
    ConditionCheck,
    GroupSummary,
    KlinkenbergResult,
    QuantityChange,
    RunSummary,
    UncertaintyBudget,
    UncertaintyComponent,
)
from gasperm.uncertainty import coverage_factor

__all__ = [
    "ComparisonError",
    "MeasurementGroup",
    "PLUG_SYMBOLS",
    "compare_groups",
    "compare_quantity",
    "klinkenberg_uncertainty_split",
    "pair_components",
    "combine_pairings",
    "slippage_explained_ratio",
]

#: Inputs that belong to the **core plug** rather than to the rig. They cancel
#: between two measurements only when it is the same plug *and* the value was
#: carried over rather than measured again -- see the module docstring.
PLUG_SYMBOLS: frozenset[str] = frozenset({"L", "d", "phi"})

#: Relative tolerance for deciding two component values are "the same number".
#: Caliper readings are stored to far better than this, so a genuine
#: re-measurement is unambiguous while float round-trips through YAML are not
#: mistaken for one.
_VALUE_TOLERANCE = 1e-9


class ComparisonError(ValueError):
    """Two results that cannot honestly be compared, with the reason."""


@dataclass(frozen=True)
class MeasurementGroup:
    """One side of a comparison: the runs, what they reduced to, and their state.

    Assembled by the caller (the CLI) from stored runs; this module never
    touches the filesystem.
    """

    label: str
    sample_id: str | None
    summaries: tuple[RunSummary, ...]
    klinkenberg: KlinkenbergResult | None = None
    porosity_fraction: float | None = None
    porosity_uncertainty: float | None = None
    #: How each run obtained P2, from ``storage.downstream_convention``. P2
    #: sets both k and the mean pressure it is quoted at, so two campaigns that
    #: obtained it differently are not on the same axis.
    downstream_conventions: tuple[str, ...] = ()

    @property
    def budgets(self) -> tuple[UncertaintyBudget, ...]:
        """Every run budget on this side, in run order."""
        return tuple(s.uncertainty for s in self.summaries if s.uncertainty is not None)

    @property
    def methods(self) -> set[str]:
        return {s.method for s in self.summaries}

    @property
    def gases(self) -> set[str]:
        return {s.gas_name for s in self.summaries}

    @property
    def flowmeters(self) -> set[str]:
        return {
            s.metadata.flowmeter
            for s in self.summaries
            if s.metadata is not None and s.metadata.flowmeter
        }

    @property
    def mean_pressures_atm(self) -> tuple[float, ...]:
        return tuple(s.mean_pressure_atm for s in self.summaries)

    def geometry_values(self) -> dict[str, float]:
        """``{'L': cm, 'd': cm}`` as recorded, for the re-measurement check."""
        for summary in self.summaries:
            if summary.metadata is None:
                continue
            return {"L": summary.metadata.length_cm, "d": summary.metadata.diameter_cm}
        return {}


# --------------------------------------------------------------------------
# Pairing the budgets
# --------------------------------------------------------------------------


def _components_by_symbol(
    budget: UncertaintyBudget | None,
) -> dict[str, UncertaintyComponent]:
    if budget is None:
        return {}
    return {component.symbol: component for component in budget.components}


def _representative(
    budgets: Sequence[UncertaintyBudget], symbol: str
) -> UncertaintyComponent | None:
    """One component standing for a whole group.

    A group is several runs at different mean pressures, so a pressure-derived
    component genuinely differs between them. The **root-mean-square** of the
    contributions is taken, which is what the series' common systematic
    propagates into an intercept fitted across it, and the largest ``u_rel`` is
    kept for reporting so a summary never understates any single run.
    """
    found = [c for b in budgets for c in b.components if c.symbol == symbol]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    rms = math.sqrt(
        sum(c.relative_contribution**2 for c in found) / len(found)
    )
    reference = max(found, key=lambda c: abs(c.relative_contribution))
    return reference.model_copy(update={"relative_contribution": rms})


def _values_match(left: float, right: float) -> bool:
    if left == right:
        return True
    scale = max(abs(left), abs(right))
    return scale > 0.0 and abs(left - right) <= _VALUE_TOLERANCE * scale


def _is_shared(
    symbol: str,
    before: UncertaintyComponent,
    after: UncertaintyComponent,
    *,
    same_sample: bool,
) -> tuple[bool, str]:
    """Whether one physical error produced both readings, and why.

    The reason string is not decoration: it is the audit trail for a claim that
    an uncertainty cancelled, and it is printed with the result.
    """
    if before.evaluation_type == "A" or after.evaluation_type == "A":
        return False, "Type A -- statistical scatter is an independent draw each run"
    if before.source != after.source:
        return False, f"different source ({before.source!r} vs {after.source!r})"
    if symbol in PLUG_SYMBOLS:
        if not same_sample:
            return False, "different plug, so this is a different physical quantity"
        if not _values_match(before.value, after.value):
            return (
                False,
                "same plug but the value changed, so it was measured again -- "
                "two independent readings do not cancel",
            )
        return True, "same plug, same recorded value -- one error, common to both"
    return True, f"same instrument/model ({before.source or symbol})"


def pair_components(
    before: Sequence[UncertaintyBudget],
    after: Sequence[UncertaintyBudget],
    *,
    same_sample: bool,
    force_independent: Iterable[str] = (),
) -> list[ComponentPairing]:
    """Match the two sides' budget components and decide which cancel.

    Args:
        before: Every run budget on the baseline side.
        after: Every run budget on the comparison side.
        same_sample: Whether both sides measured the same core plug.
        force_independent: Symbols to treat as independent whatever the
            components say -- used when a condition check found the two sides
            were not, in fact, measured with the same instrument.

    Returns:
        One pairing per symbol seen on either side, each carrying its variance
        contribution to the **ratio**.
    """
    blocked = set(force_independent)
    symbols: list[str] = []
    for budgets in (before, after):
        for budget in budgets:
            for component in budget.components:
                if component.symbol not in symbols:
                    symbols.append(component.symbol)

    pairings: list[ComponentPairing] = []
    for symbol in symbols:
        left = _representative(before, symbol)
        right = _representative(after, symbol)
        contribution_before = abs(left.relative_contribution) if left else 0.0
        contribution_after = abs(right.relative_contribution) if right else 0.0

        if left is None or right is None:
            only = left or right
            pairings.append(
                ComponentPairing(
                    symbol=symbol,
                    name=only.name,
                    shared=False,
                    reason=f"present only on the {'before' if left else 'after'} side",
                    type_a=only.evaluation_type == "A",
                    relative_contribution_before=contribution_before,
                    relative_contribution_after=contribution_after,
                    variance_contribution=contribution_before**2 + contribution_after**2,
                    degrees_of_freedom_before=(
                        left.degrees_of_freedom if left else math.inf
                    ),
                    degrees_of_freedom_after=(
                        right.degrees_of_freedom if right else math.inf
                    ),
                )
            )
            continue

        shared, reason = _is_shared(symbol, left, right, same_sample=same_sample)
        if shared and symbol in blocked:
            shared, reason = False, "conditions differed between the two campaigns"

        if shared:
            # Perfect correlation: the residue is the DIFFERENCE of the two
            # contributions, which is zero when the readings match and grows
            # exactly as far as they drift apart. The two sensitivities are the
            # same exponent of the same measurand, so subtracting magnitudes is
            # subtracting signed contributions.
            variance = (contribution_before - contribution_after) ** 2
        else:
            variance = contribution_before**2 + contribution_after**2

        pairings.append(
            ComponentPairing(
                symbol=symbol,
                name=left.name,
                shared=shared,
                reason=reason,
                type_a=left.evaluation_type == "A" or right.evaluation_type == "A",
                relative_contribution_before=contribution_before,
                relative_contribution_after=contribution_after,
                variance_contribution=variance,
                degrees_of_freedom_before=left.degrees_of_freedom,
                degrees_of_freedom_after=right.degrees_of_freedom,
            )
        )
    return pairings


def _welch(variance: float, terms: Sequence[tuple[float, float]]) -> float:
    """Welch-Satterthwaite effective degrees of freedom, GUM annex G.4.

    ``terms`` are ``(variance, dof)`` pairs. A term with infinite dof (a pure
    Type B input) adds nothing to the denominator, which is what lets an
    all-Type-B budget report infinite dof and fall back to the normal quantile.
    """
    if variance <= 0.0:
        return math.inf
    denominator = 0.0
    for term_variance, dof in terms:
        if term_variance <= 0.0 or not math.isfinite(dof) or dof <= 0.0:
            continue
        denominator += term_variance**2 / dof
    if denominator <= 0.0:
        return math.inf
    return variance**2 / denominator


def combine_pairings(pairings: Sequence[ComponentPairing]) -> tuple[float, float]:
    """``(u_rel, effective_dof)`` of the ratio, per GUM 5 and annex G.

    Welch-Satterthwaite over the surviving terms only: a component that
    cancelled contributes no variance and therefore no degrees of freedom
    either, which is the whole reason a paired comparison can be far better
    determined than its inputs.
    """
    variance = sum(max(p.variance_contribution, 0.0) for p in pairings)
    u_rel = math.sqrt(variance)
    if u_rel <= 0.0:
        return 0.0, math.inf
    return u_rel, _welch(
        variance, [term for p in pairings for term in p.welch_terms]
    )


# --------------------------------------------------------------------------
# Turning that into a reported change
# --------------------------------------------------------------------------


def compare_quantity(
    name: str,
    symbol: str,
    unit: str,
    before: float,
    after: float,
    *,
    relative_uncertainty: float,
    effective_dof: float = math.inf,
    coverage_probability: float = 0.95,
    paired: bool,
    notes: Sequence[str] = (),
) -> QuantityChange:
    """Package a before/after pair into a reported change with a verdict.

    ``relative_uncertainty`` is that of the **ratio**, already combined by
    :func:`combine_pairings`. The verdict compares the observed change against
    its own expanded uncertainty, so a change is called significant only when
    it exceeds what the measurement could have produced by chance at the stated
    level of confidence.
    """
    k = coverage_factor(effective_dof, coverage_probability)
    ratio = after / before if before else math.inf
    difference = after - before
    u_ratio = relative_uncertainty
    expanded_ratio = k * u_ratio

    percent_change = (ratio - 1.0) * 100.0 if math.isfinite(ratio) else math.inf
    # The smallest change this comparison could have resolved. A null result is
    # only informative alongside it.
    minimum_detectable = expanded_ratio * 100.0

    significant = (
        math.isfinite(ratio)
        and expanded_ratio > 0.0
        and abs(ratio - 1.0) > expanded_ratio
    )
    if math.isfinite(ratio) and expanded_ratio == 0.0:
        significant = ratio != 1.0

    return QuantityChange(
        name=name,
        symbol=symbol,
        unit=unit,
        before=before,
        after=after,
        difference=difference,
        ratio=ratio,
        percent_change=percent_change,
        relative_standard_uncertainty=u_ratio,
        relative_expanded_uncertainty=expanded_ratio,
        standard_uncertainty=abs(before) * u_ratio if math.isfinite(ratio) else math.inf,
        coverage_factor=k,
        coverage_probability=coverage_probability,
        effective_degrees_of_freedom=effective_dof,
        minimum_detectable_percent=minimum_detectable,
        significant=significant,
        paired=paired,
        notes=list(notes),
    )


# --------------------------------------------------------------------------
# k_L: separating what the fit sees from what it cannot
# --------------------------------------------------------------------------


def klinkenberg_uncertainty_split(
    result: KlinkenbergResult, budgets: Sequence[UncertaintyBudget]
) -> tuple[float, float]:
    """``(u_rel from scatter alone, u_rel systematic)`` for one series' ``k_L``.

    The two halves have to be separated because only one of them cancels from a
    paired ratio. Inputs common to every point -- plug geometry, the
    transducers, the meter, the viscosity model -- move all of them together, so
    they slide the intercept without changing the fit residuals at all. Since
    ``k_g = k_L (1 + b/P)`` is linear in ``k_L``, a common relative error ``e``
    scales the intercept by exactly ``(1+e)`` and leaves ``b`` untouched, which
    is why the systematic half is also precisely the half that cancels.

    **A weighted fit's intercept standard error is not the scatter half.** When
    every point carried an uncertainty, ``fit_klinkenberg`` weighted by
    ``1/u^2`` and its intercept standard error propagated each point's *whole*
    uncertainty -- systematic included, and treated as independent. Handing that
    back while also adding the systematic separately would charge the same error
    twice, and treating it as independent understates it in the first place. So
    it is scaled to the genuinely independent share of each point's budget: the
    Type A part, which is what actually produces scatter about the line.

    An unweighted fit needs no such correction. Its intercept standard error
    comes from the residuals, which are already a purely random estimate.
    """
    intercept = result.intercept
    if intercept == 0.0 or not math.isfinite(intercept):
        return math.inf, math.inf

    fit_rel = 0.0
    if result.intercept_stderr is not None and math.isfinite(result.intercept_stderr):
        fit_rel = abs(result.intercept_stderr / intercept)

    random_terms: list[float] = []
    systematic_terms: list[float] = []
    for budget in budgets:
        random_variance = sum(
            c.relative_contribution**2
            for c in budget.components
            if c.evaluation_type == "A"
        )
        systematic_variance = sum(
            c.relative_contribution**2
            for c in budget.components
            if c.evaluation_type != "A"
        )
        random_terms.append(math.sqrt(random_variance))
        systematic_terms.append(math.sqrt(systematic_variance))

    if result.weighted and random_terms:
        # Var(intercept) of a weighted fit scales as the square of a uniform
        # rescaling of the weights' uncertainties, so the standard error scales
        # linearly with it.
        total = math.hypot(_rms(random_terms), _rms(systematic_terms))
        if total > 0.0:
            fit_rel *= _rms(random_terms) / total

    return fit_rel, _rms(systematic_terms)


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(v**2 for v in values) / len(values)) if values else 0.0


def slippage_explained_ratio(
    slippage_factor_atm: float, before_pressure_atm: float, after_pressure_atm: float
) -> float:
    """How much of an apparent ``k_g`` ratio is just a mean-pressure mismatch.

    From ``k_g(P) = k_L (1 + b/P)``, two runs at different mean pressures
    differ by ``(1 + b/P_after) / (1 + b/P_before)`` **even when the rock is
    unchanged**. Quoting this alongside an observed ratio is the difference
    between "permeability fell 12%" and "permeability fell 12%, of which 3% is
    the pressure mismatch".
    """
    if before_pressure_atm <= 0.0 or after_pressure_atm <= 0.0:
        raise ValueError("mean pressures must be positive")
    return (1.0 + slippage_factor_atm / after_pressure_atm) / (
        1.0 + slippage_factor_atm / before_pressure_atm
    )


# --------------------------------------------------------------------------
# Condition matching
# --------------------------------------------------------------------------

#: Differences that make a comparison meaningless rather than merely noisier.
#: Each maps to the symbols whose cancellation it also invalidates.
_BLOCKING = {
    "method": (),
    "gas": ("mu", "c_g"),
    "downstream_pressure": (),
}

#: Differences that are survivable but must void a cancellation claim.
_VOIDS_SHARING = {
    "flowmeter": ("Q", "P_ref"),
    "vessels": ("V1", "V2"),
}


def _check(
    key: str, label: str, before: object, after: object, *, advice: str = ""
) -> ConditionCheck:
    matched = before == after
    return ConditionCheck(
        key=key,
        label=label,
        before=_render(before),
        after=_render(after),
        matched=matched,
        blocking=(not matched) and key in _BLOCKING,
        voids_symbols=list(() if matched else _VOIDS_SHARING.get(key, ())),
        advice=advice if not matched else "",
    )


def _render(value: object) -> str:
    if isinstance(value, (set, frozenset)):
        return ", ".join(sorted(str(v) for v in value)) or "(none)"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value) if value not in (None, "") else "(unset)"


def check_conditions(
    before: MeasurementGroup, after: MeasurementGroup
) -> list[ConditionCheck]:
    """Everything that had to be the same for the difference to mean anything.

    This is the retroactive form of running both campaigns from one protocol:
    it cannot make the conditions match, but it can stop a mismatch being
    reported as a result.
    """
    checks = [
        _check(
            "method", "measurement method", before.methods, after.methods,
            advice="Steady-state and pulse-decay permeabilities carry a systematic "
                   "offset relative to each other; a change measured across the two "
                   "is that offset plus whatever the rock did.",
        ),
        _check(
            "gas", "working gas", before.gases, after.gases,
            advice="Viscosity and slippage both depend on the gas, so k_g is not "
                   "comparable across a change of it.",
        ),
        _check(
            "downstream_pressure", "how P2 was obtained",
            set(before.downstream_conventions), set(after.downstream_conventions),
            advice="P2 sets both the permeability and the mean pressure it is "
                   "quoted at, so a change of convention moves the result and the "
                   "axis it sits on at the same time.",
        ),
        _check(
            "flowmeter", "flowmeter", before.flowmeters, after.flowmeters,
            advice="A different meter has a different calibration error, so the "
                   "meter no longer cancels from the ratio -- it is now charged to "
                   "the comparison in full, on both sides.",
        ),
    ]

    geometry_before = before.geometry_values()
    geometry_after = after.geometry_values()
    if geometry_before and geometry_after:
        same = all(
            _values_match(geometry_before[key], geometry_after[key])
            for key in ("L", "d")
        )
        checks.append(
            ConditionCheck(
                key="geometry",
                label="plug geometry (L, d)",
                before=f"{geometry_before['L']:.4f} x {geometry_before['d']:.4f} cm",
                after=f"{geometry_after['L']:.4f} x {geometry_after['d']:.4f} cm",
                matched=same,
                blocking=False,
                voids_symbols=[],
                advice=(
                    ""
                    if same
                    else "The plug was measured again between campaigns. Caliper "
                         "error therefore no longer cancels: it is two independent "
                         "readings, and both are charged to the comparison."
                ),
            )
        )
    return checks


# --------------------------------------------------------------------------
# The whole comparison
# --------------------------------------------------------------------------


def compare_groups(
    before: MeasurementGroup,
    after: MeasurementGroup,
    *,
    coverage_probability: float = 0.95,
    paired: bool | None = None,
    allow_mismatched_conditions: bool = False,
    pressure_tolerance: float = 0.05,
) -> ComparisonResult:
    """Compare two sets of runs and report every difference, with its uncertainty.

    Args:
        before: The baseline group.
        after: The group to compare against it.
        coverage_probability: Level of confidence for every expanded uncertainty.
        paired: Force the same-plug treatment on or off. ``None`` decides from
            the sample ids, which is right in every ordinary case.
        allow_mismatched_conditions: Proceed despite a blocking condition
            difference. The differences are still reported and still flagged.
        pressure_tolerance: Relative window for calling two runs' mean
            pressures the same point, when matching them up run by run.

    Raises:
        ComparisonError: a blocking condition differs and was not allowed, or
            neither side has anything to compare.
    """
    if not before.summaries or not after.summaries:
        raise ComparisonError(
            "Both sides need at least one confirmed run. "
            f"{before.label!r} has {len(before.summaries)}, "
            f"{after.label!r} has {len(after.summaries)}."
        )

    same_sample = (
        paired
        if paired is not None
        else bool(
            before.sample_id and after.sample_id and before.sample_id == after.sample_id
        )
    )

    checks = check_conditions(before, after)
    blocking = [c for c in checks if c.blocking]
    if blocking and not allow_mismatched_conditions:
        detail = "\n".join(
            f"  {c.label}: {c.before}  ->  {c.after}\n    {c.advice}" for c in blocking
        )
        raise ComparisonError(
            "These two campaigns were not run under comparable conditions:\n"
            f"{detail}\n"
            "The difference between them is that mismatch plus whatever the sample "
            "did, and nothing here can separate the two. Re-run under matched "
            "conditions, or pass --allow-mismatched-conditions to report it anyway."
        )

    voided: set[str] = set()
    for check in checks:
        voided.update(check.voids_symbols)
    if not same_sample:
        voided.update(PLUG_SYMBOLS)

    pairings = pair_components(
        before.budgets, after.budgets, same_sample=same_sample, force_independent=voided
    )

    changes: list[QuantityChange] = []
    warnings: list[str] = []

    # -- the headline: liquid-equivalent permeability ----------------------
    if before.klinkenberg is not None and after.klinkenberg is not None:
        changes.append(
            _compare_klinkenberg_intercept(
                before, after, pairings,
                same_sample=same_sample,
                coverage_probability=coverage_probability,
            )
        )
        changes.append(
            _compare_slippage(
                before, after, coverage_probability=coverage_probability,
                paired=same_sample,
            )
        )
    else:
        missing = [
            group.label
            for group in (before, after)
            if group.klinkenberg is None
        ]
        warnings.append(
            "No Klinkenberg fit for " + " and ".join(missing) + ", so the comparison "
            "is of apparent permeability k_g at matched mean pressures rather than of "
            "the liquid-equivalent k_L. k_g carries the slippage term, so it is only "
            "comparable where the mean pressures line up -- which is why the "
            "per-pressure table below reports how far apart they were."
        )

    # -- apparent permeability, run by matched run ------------------------
    matched_points, unmatched = _match_by_pressure(before, after, pressure_tolerance)
    for pair_before, pair_after in matched_points:
        changes.append(
            _compare_matched_runs(
                pair_before, pair_after, pairings, before, after,
                same_sample=same_sample,
                coverage_probability=coverage_probability,
            )
        )
    if unmatched:
        warnings.append(
            f"{len(unmatched)} run(s) had no counterpart within "
            f"{pressure_tolerance:.0%} of their mean pressure and were left out of "
            "the per-pressure table: "
            + ", ".join(f"{p:.3g} atm" for p in unmatched)
        )

    # -- porosity ---------------------------------------------------------
    if before.porosity_fraction is not None and after.porosity_fraction is not None:
        changes.append(
            _compare_porosity(before, after, coverage_probability=coverage_probability,
                              paired=same_sample)
        )

    if not same_sample:
        warnings.append(
            "These are different plugs, so nothing about the rock cancels: the "
            "geometry, the porosity and the plug-to-plug variability are all charged "
            "to the comparison. Only the rig-level inputs -- transducers, meter, gas "
            "model -- still cancel, because both were measured on the same bench."
        )

    return ComparisonResult(
        before=_group_summary(before),
        after=_group_summary(after),
        paired=same_sample,
        coverage_probability=coverage_probability,
        changes=changes,
        conditions=checks,
        component_pairings=pairings,
        warnings=warnings,
    )


def _compare_klinkenberg_intercept(
    before: MeasurementGroup,
    after: MeasurementGroup,
    pairings: Sequence[ComponentPairing],
    *,
    same_sample: bool,
    coverage_probability: float,
) -> QuantityChange:
    """k_L before against k_L after, with the systematic half cancelled out."""
    fit_before, sys_before = klinkenberg_uncertainty_split(
        before.klinkenberg, before.budgets
    )
    fit_after, sys_after = klinkenberg_uncertainty_split(after.klinkenberg, after.budgets)

    # Only the SYSTEMATIC pairings belong here. The Type A ones are the point
    # scatter, and the point scatter is already what the intercept's standard
    # error above describes -- adding it again would count the same
    # repeatability twice, once through the fit and once beside it.
    systematic = [p for p in pairings if not p.type_a]
    systematic_rel, systematic_dof = combine_pairings(systematic)
    u_rel = math.sqrt(fit_before**2 + fit_after**2 + systematic_rel**2)

    terms = [
        (fit_before**2, _fit_dof(before.klinkenberg)),
        (fit_after**2, _fit_dof(after.klinkenberg)),
        (systematic_rel**2, systematic_dof),
    ]
    dof = _welch(u_rel**2, terms)

    notes = [
        f"point scatter propagates {fit_before:.3%} (before) and {fit_after:.3%} "
        f"(after) into the intercept; the systematic inputs that did NOT cancel add "
        f"{systematic_rel:.3%}.",
    ]
    absolute_before = math.hypot(fit_before, sys_before)
    absolute_after = math.hypot(fit_after, sys_after)
    notes.append(
        f"For reference the ABSOLUTE uncertainties are {absolute_before:.2%} and "
        f"{absolute_after:.2%}; the ratio is better determined than either because "
        "the shared inputs cancel."
    )
    return compare_quantity(
        "liquid-equivalent permeability", "k_L", "darcy",
        before.klinkenberg.liquid_permeability_darcy,
        after.klinkenberg.liquid_permeability_darcy,
        relative_uncertainty=u_rel,
        effective_dof=dof,
        coverage_probability=coverage_probability,
        paired=same_sample,
        notes=notes,
    )


def _fit_dof(result: KlinkenbergResult) -> float:
    """``n - 2`` for a two-parameter fit; infinite when it cannot be formed."""
    return float(result.point_count - 2) if result.point_count > 2 else math.inf


def _compare_slippage(
    before: MeasurementGroup,
    after: MeasurementGroup,
    *,
    coverage_probability: float,
    paired: bool,
) -> QuantityChange:
    """The slippage factor, which is a second observable and often the sharper one.

    ``b`` depends on the pore throat size relative to the gas mean free path, so
    it can move before ``k`` does when pore structure changes. It is also
    immune to everything that merely scales ``k_g``: a calibration error common
    to the whole series cancels out of ``b`` exactly, which is why its
    uncertainty here comes from the two fits alone.
    """
    b_before = before.klinkenberg.slippage_factor_atm
    b_after = after.klinkenberg.slippage_factor_atm
    u_before = _relative_or_inf(
        before.klinkenberg.slippage_factor_standard_uncertainty_atm, b_before
    )
    u_after = _relative_or_inf(
        after.klinkenberg.slippage_factor_standard_uncertainty_atm, b_after
    )
    u_rel = math.hypot(u_before, u_after)
    dof = _welch(u_rel**2, [
        (u_before**2, _fit_dof(before.klinkenberg)),
        (u_after**2, _fit_dof(after.klinkenberg)),
    ])
    return compare_quantity(
        "gas slippage factor", "b", "atm", b_before, b_after,
        relative_uncertainty=u_rel,
        effective_dof=dof,
        coverage_probability=coverage_probability,
        paired=paired,
        notes=[
            "b is unaffected by any error that merely scales the whole series, so "
            "its uncertainty here is the two regressions' own and nothing else.",
        ],
    )


def _relative_or_inf(uncertainty: float | None, value: float) -> float:
    if uncertainty is None or not math.isfinite(uncertainty) or value == 0.0:
        return math.inf
    return abs(uncertainty / value)


def _match_by_pressure(
    before: MeasurementGroup, after: MeasurementGroup, tolerance: float
) -> tuple[list[tuple[RunSummary, RunSummary]], list[float]]:
    """Pair up runs whose mean pressures agree, greedily and closest-first."""
    remaining = list(after.summaries)
    matched: list[tuple[RunSummary, RunSummary]] = []
    unmatched: list[float] = []
    for summary in before.summaries:
        candidates = [
            (abs(other.mean_pressure_atm - summary.mean_pressure_atm), index)
            for index, other in enumerate(remaining)
        ]
        if not candidates:
            unmatched.append(summary.mean_pressure_atm)
            continue
        gap, index = min(candidates)
        if gap > tolerance * summary.mean_pressure_atm:
            unmatched.append(summary.mean_pressure_atm)
            continue
        matched.append((summary, remaining.pop(index)))
    unmatched.extend(other.mean_pressure_atm for other in remaining)
    return matched, unmatched


def _compare_matched_runs(
    before_run: RunSummary,
    after_run: RunSummary,
    group_pairings: Sequence[ComponentPairing],
    before: MeasurementGroup,
    after: MeasurementGroup,
    *,
    same_sample: bool,
    coverage_probability: float,
) -> QuantityChange:
    """One matched pressure point, priced from just those two runs' budgets."""
    voided: set[str] = set() if same_sample else set(PLUG_SYMBOLS)
    pairings = pair_components(
        [before_run.uncertainty] if before_run.uncertainty else [],
        [after_run.uncertainty] if after_run.uncertainty else [],
        same_sample=same_sample,
        force_independent=voided,
    )
    u_rel, dof = combine_pairings(pairings)
    if not pairings:
        u_rel, dof = math.inf, math.inf

    notes = []
    gap = after_run.mean_pressure_atm - before_run.mean_pressure_atm
    fit = after.klinkenberg or before.klinkenberg
    if fit is not None and abs(gap) > 0.0:
        try:
            explained = slippage_explained_ratio(
                fit.slippage_factor_atm,
                before_run.mean_pressure_atm,
                after_run.mean_pressure_atm,
            )
        except ValueError:
            explained = None
        if explained is not None and math.isfinite(explained):
            notes.append(
                f"mean pressure moved {before_run.mean_pressure_atm:.3g} -> "
                f"{after_run.mean_pressure_atm:.3g} atm, which alone accounts for "
                f"{(explained - 1.0) * 100.0:+.2f}% of the change through slippage."
            )
    return compare_quantity(
        f"apparent permeability at {before_run.mean_pressure_atm:.3g} atm",
        "k_g", "darcy",
        before_run.permeability_darcy, after_run.permeability_darcy,
        relative_uncertainty=u_rel,
        effective_dof=dof,
        coverage_probability=coverage_probability,
        paired=same_sample,
        notes=notes,
    )


def _compare_porosity(
    before: MeasurementGroup,
    after: MeasurementGroup,
    *,
    coverage_probability: float,
    paired: bool,
) -> QuantityChange:
    """Porosity, whose uncertainty is whatever was entered with it.

    Deliberately not assumed to cancel: porosity is normally a separate
    instrument's result, so the two values are two independent measurements
    even on the same plug.
    """
    u_before = before.porosity_uncertainty or 0.0
    u_after = after.porosity_uncertainty or 0.0
    phi_before = before.porosity_fraction
    phi_after = after.porosity_fraction
    u_rel = math.hypot(
        u_before / phi_before if phi_before else math.inf,
        u_after / phi_after if phi_after else math.inf,
    )
    notes = []
    if not (before.porosity_uncertainty and after.porosity_uncertainty):
        notes.append(
            "sample.porosity_uncertainty is unrecorded on at least one side, so the "
            "change is reported without one -- record it to get a verdict."
        )
    return compare_quantity(
        "porosity", "phi", "fraction", phi_before, phi_after,
        relative_uncertainty=u_rel,
        effective_dof=math.inf,
        coverage_probability=coverage_probability,
        paired=paired,
        notes=notes,
    )


def _group_summary(group: MeasurementGroup) -> GroupSummary:
    fit = group.klinkenberg
    return GroupSummary(
        label=group.label,
        sample_id=group.sample_id,
        run_count=len(group.summaries),
        mean_pressures_atm=list(group.mean_pressures_atm),
        methods=sorted(group.methods),
        gases=sorted(group.gases),
        flowmeters=sorted(group.flowmeters),
        liquid_permeability_darcy=fit.liquid_permeability_darcy if fit else None,
        slippage_factor_atm=fit.slippage_factor_atm if fit else None,
        r_squared=fit.r_squared if fit else None,
        porosity_fraction=group.porosity_fraction,
        started_at=min((s.started_at for s in group.summaries), default=None),
        ended_at=max((s.ended_at for s in group.summaries), default=None),
    )
