"""Uncertainty propagation per ISO/IEC Guide 98-3 (the GUM).

The measurand is the apparent gas permeability::

    k = 2 Q P_ref mu L / (A (P1^2 - P2^2)),   A = pi d^2 / 4

Because that is a product of powers, the GUM law of propagation of uncertainty
(equation 10) takes a particularly clean form: the *relative* sensitivity
coefficient of each input is simply its exponent,

    c_i^rel = (x_i / k) (dk / dx_i)

so ``u_c(k)/k`` is a quadrature sum of ``|c_i^rel| u(x_i)/x_i``. The exponents
are +1 for Q, P_ref, mu and L; -2 for the diameter (area goes as d^2, which is
why the caliper term is usually the largest in the budget); and, for the two
pressures,

    c_P1 = -2 P1^2 / (P1^2 - P2^2)      c_P2 = +2 P2^2 / (P1^2 - P2^2)

which blow up as the differential closes -- the quantitative statement of why
a low-differential measurement is a bad measurement.

**Correlation.** P1 and P2 usually come from two units of the same transducer
model calibrated against the same reference, so their errors are correlated.
GUM equation (13) requires the covariance term, and because the two enter with
opposite signs a positive correlation *reduces* the combined uncertainty.
Ignoring it is conservative but wrong; ``pressure_calibration.correlation``
makes it explicit.

**Type A vs Type B.** The Type A term is the standard deviation of the mean of
the steady-state window -- the run's own random scatter. The Type B terms come
from instrument specifications and represent systematic calibration error.
Treating them as independent is the standard treatment and is what the
separation of the two evaluation types is for; it does mean the random part of
a transducer's specification is counted twice, which errs high.

**Neglected.** Viscosity's weak pressure dependence: for common gases a few
atm from ambient, ``d ln mu / d ln P`` is order 1e-3, far below the viscosity
model's own uncertainty. Temperature's effect *is* included, through
``d ln mu / d ln T``.

Hardware-free: everything here takes plain numbers and config objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from gasperm import units
from gasperm.config.hardware import HardwareConfig, PressureChannelConfig
from gasperm.config.run import RunConfig, UncertaintyReportConfig
from gasperm.models import SampleGeometry, UncertaintyBudget, UncertaintyComponent

#: Above this, the Welch-Satterthwaite result is reported as infinite: the
#: Student-t factor is already within rounding of the normal quantile.
_EFFECTIVE_DOF_INFINITE = 1.0e6

__all__ = [
    "MeasurementPoint",
    "pressure_sensitivities",
    "coverage_factor",
    "combine_budget",
    "build_budget",
    "PulseDecayPoint",
    "build_pulse_decay_budget",
]


@dataclass(frozen=True)
class MeasurementPoint:
    """The steady-state means the budget is evaluated at, in internal units."""

    permeability_darcy: float
    inlet_pressure_atm: float
    #: P2 as it entered the Darcy equation -- the outlet transducer, or the
    #: value the run supplied.
    downstream_pressure_atm: float
    flow_cm3_s: float
    reference_pressure_atm: float
    viscosity_cp: float
    temperature_c: float


def pressure_sensitivities(
    inlet_pressure_atm: float, outlet_pressure_atm: float
) -> tuple[float, float]:
    """Relative sensitivity coefficients of ``k`` to P1 and P2.

    Returns:
        ``(c_P1, c_P2)``, dimensionless. ``c_P1`` is negative and ``c_P2``
        positive; both diverge as ``P1 -> P2``.

    Raises:
        ValueError: the pressures are equal, so ``k`` is not defined.
    """
    denominator = inlet_pressure_atm**2 - outlet_pressure_atm**2
    if denominator == 0.0:
        raise ValueError(
            "inlet and outlet pressures are equal, so the sensitivity coefficients "
            "are undefined"
        )
    return (
        -2.0 * inlet_pressure_atm**2 / denominator,
        2.0 * outlet_pressure_atm**2 / denominator,
    )


def coverage_factor(effective_dof: float, coverage_probability: float) -> float:
    """Coverage factor ``k`` for a level of confidence, per GUM annex G.

    Student-t at the effective degrees of freedom, falling back to the normal
    quantile when the degrees of freedom are infinite (all-Type-B budgets).
    """
    if not 0.0 < coverage_probability < 1.0:
        raise ValueError(
            f"coverage_probability must be in (0, 1), got {coverage_probability}"
        )
    from scipy import stats

    tail = 1.0 - (1.0 - coverage_probability) / 2.0
    if not math.isfinite(effective_dof) or effective_dof > 1e6:
        return float(stats.norm.ppf(tail))
    # Below 1 dof the t distribution has no usable variance; clamp so a budget
    # built from a single sample still reports something finite and obviously
    # large rather than raising.
    return float(stats.t.ppf(tail, max(effective_dof, 1.0)))


def _pressure_relative_uncertainty(
    channel: PressureChannelConfig,
    absolute_pressure_atm: float,
    run: RunConfig,
    daq_relative: float,
) -> tuple[float, float]:
    """``(u_rel, dof)`` for one pressure channel at the given absolute pressure.

    The specification applies to what the *transducer* reads. For a gauge
    channel that is the gauge value, and the absolute pressure the equation
    needs is that reading plus the ambient reference -- so the ambient value's
    own uncertainty joins in quadrature. (If both channels are gauge they share
    that ambient error; express the resulting correlation through
    ``pressure_calibration.correlation``.) The relative uncertainty is taken
    against the absolute pressure, which is what enters the equation.
    """
    reading_atm = absolute_pressure_atm
    if channel.reading_type == "gauge":
        reading_atm = absolute_pressure_atm - run.atmospheric_pressure_atm
    reading_in_unit = units.from_atm(reading_atm, channel.unit)

    u_in_unit = channel.uncertainty.standard_uncertainty(reading_in_unit, channel.full_scale)
    # Pressure units are pure scalings, so a difference converts like a value.
    u_atm = units.to_atm(u_in_unit, channel.unit)

    if channel.reading_type == "gauge":
        ambient_spec = run.atmospheric_pressure_uncertainty
        u_ambient_atm = units.to_atm(
            ambient_spec.standard_uncertainty(
                run.atmospheric_pressure, abs(run.atmospheric_pressure)
            ),
            run.atmospheric_pressure_unit,
        )
        u_atm = math.hypot(u_atm, u_ambient_atm)

    if absolute_pressure_atm == 0.0:
        return math.inf, channel.uncertainty.dof
    u_relative = u_atm / abs(absolute_pressure_atm)
    # The DAQ's own voltage error rides on the same signal.
    return math.hypot(u_relative, daq_relative), channel.uncertainty.dof


def _component(
    *,
    name: str,
    symbol: str,
    evaluation_type: str,
    value: float,
    unit: str,
    relative_uncertainty: float,
    sensitivity: float,
    dof: float,
    source: str,
) -> UncertaintyComponent:
    return UncertaintyComponent(
        name=name,
        symbol=symbol,
        evaluation_type=evaluation_type,
        value=value,
        unit=unit,
        standard_uncertainty=abs(value) * relative_uncertainty,
        relative_standard_uncertainty=relative_uncertainty,
        relative_sensitivity=sensitivity,
        relative_contribution=abs(sensitivity * relative_uncertainty),
        degrees_of_freedom=dof,
        source=source,
    )


@dataclass(frozen=True)
class _Correlation:
    """One covariance term between two inputs that share an error source.

    GUM 5.2: the combined variance gains ``2 c_a c_b r u_a u_b`` for each
    correlated pair. Whether that raises or lowers the total depends on the
    signs, and the two measurands here differ in exactly that way -- P1 and P2
    enter the Darcy equation with opposite signs, so shared transducer error
    partly cancels, while both pulse-decay vessel volumes enter positively, so
    a shared calibration error adds.
    """

    label: str
    sensitivity_a: float
    sensitivity_b: float
    relative_uncertainty_a: float
    relative_uncertainty_b: float
    coefficient: float
    #: Why this pair is correlated and which way the term goes, appended to the
    #: budget note. Generic arithmetic cannot say why, and "why" is the part an
    #: operator needs when a stated correlation turns out to be wrong.
    explanation: str = ""

    @property
    def relative_variance(self) -> float:
        return (
            2.0
            * self.sensitivity_a
            * self.sensitivity_b
            * self.coefficient
            * self.relative_uncertainty_a
            * self.relative_uncertainty_b
        )


def combine_budget(
    components: Sequence[UncertaintyComponent],
    *,
    value_darcy: float,
    report: UncertaintyReportConfig,
    correlations: Sequence[_Correlation] = (),
    measurand: str = "apparent gas permeability",
    notes: Sequence[str] = (),
) -> UncertaintyBudget:
    """Combine evaluated components into a reported budget.

    Pure arithmetic over the component list: quadrature sum, covariance terms,
    Welch-Satterthwaite effective degrees of freedom, and the coverage factor.
    It knows nothing about which measurand produced the components, which is
    what lets the steady-state and pulse-decay builders share it -- and sharing
    matters here, because two copies of the dof clamp and the non-positive
    variance fallback would eventually disagree.

    Args:
        components: Evaluated inputs, each carrying its relative contribution.
        value_darcy: The measurand itself, used to scale relative to absolute.
        report: Coverage settings.
        correlations: Covariance terms between pairs of the above.
        measurand: What is being reported, for the budget header.
        notes: Caller's notes; this function appends its own.

    Returns:
        The full budget, ready to print or store.
    """
    collected = list(notes)
    component_variance = sum(component.variance_share for component in components)
    variance = component_variance
    correlation_variance = 0.0

    for correlation in correlations:
        if correlation.coefficient == 0.0:
            continue
        term = correlation.relative_variance
        correlation_variance += term
        variance += term
        message = (
            f"Included the {correlation.label} covariance term at "
            f"r = {correlation.coefficient:g}; "
        )
        if correlation.explanation:
            message += correlation.explanation + " this "
        else:
            message += "this "
        message += f"{'reduces' if term < 0 else 'increases'} the combined uncertainty."
        collected.append(message)

    if correlation_variance != 0.0 and variance <= 0.0:
        variance = component_variance
        correlation_variance = 0.0
        collected.append(
            "The covariance term drove the combined variance non-positive, which "
            "means the stated correlation is too strong to be physical here; it was "
            "dropped and the uncorrelated result is reported instead."
        )

    relative_combined = math.sqrt(variance)
    combined = relative_combined * abs(value_darcy)

    # -- Welch-Satterthwaite ----------------------------------------------
    denominator = 0.0
    for component in components:
        if math.isfinite(component.degrees_of_freedom) and component.degrees_of_freedom > 0:
            denominator += component.variance_share**2 / component.degrees_of_freedom
    effective_dof = math.inf if denominator == 0.0 else variance**2 / denominator
    if effective_dof > _EFFECTIVE_DOF_INFINITE:
        # A finite-dof term with a negligible variance share produces an
        # astronomically large v_eff. Anything past this is indistinguishable
        # from infinity for the coverage factor, and reporting the raw number
        # implies a precision that is not there.
        effective_dof = math.inf

    factor = (
        report.fixed_coverage_factor
        if report.fixed_coverage_factor is not None
        else coverage_factor(effective_dof, report.coverage_probability)
    )

    return UncertaintyBudget(
        measurand=measurand,
        value_darcy=value_darcy,
        combined_standard_uncertainty_darcy=combined,
        relative_combined_standard_uncertainty=relative_combined,
        effective_degrees_of_freedom=effective_dof,
        coverage_factor=factor,
        coverage_probability=report.coverage_probability,
        expanded_uncertainty_darcy=factor * combined,
        components=list(components),
        correlation_relative_variance=correlation_variance,
        notes=collected,
    )


def build_budget(
    point: MeasurementPoint,
    geometry: SampleGeometry,
    hardware: HardwareConfig,
    run: RunConfig,
    *,
    type_a_relative: float | None = None,
    type_a_dof: float = math.inf,
    viscosity_temperature_exponent: float = 0.0,
) -> UncertaintyBudget:
    """Evaluate the full GUM budget for one measured permeability.

    Args:
        point: Steady-state means, in internal CGS units.
        geometry: Plug geometry with its caliper uncertainties.
        hardware: Instrument specifications.
        run: Coverage settings and how P2 was determined.
        type_a_relative: Relative standard uncertainty of the mean from the
            steady-state window's scatter, ``s / (sqrt(n) |mean|)``. ``None``
            omits the Type A term.
        type_a_dof: Degrees of freedom for that term, normally ``n - 1``.
        viscosity_temperature_exponent: ``d ln mu / d ln T`` at the test
            conditions, from :mod:`gasperm.gas_properties`. Zero omits the
            temperature term.

    Returns:
        The budget, with every component and the expanded uncertainty.

    Raises:
        ValueError: the pressures are equal, so no budget can be formed.
    """
    notes: list[str] = []
    components: list[UncertaintyComponent] = []

    daq_relative = hardware.uncertainty.daq_relative
    calibration = hardware.pressure_calibration

    c_p1, c_p2 = pressure_sensitivities(
        point.inlet_pressure_atm, point.downstream_pressure_atm
    )
    # A supplied downstream pressure is a stated constant, not a reading: it
    # carries the operator's own uncertainty and shares no transducer error
    # with P1, so the covariance term must not be applied to it.
    supplied_downstream = run.fixed_downstream_pressure_atm is not None

    # -- flow rate --------------------------------------------------------
    # The meter this run selected: its own range and specification, not the
    # rig's other meter's.
    flow_name, flow = hardware.resolve_flowmeter(run.flowmeter)
    flow_in_unit = units.flow_from_cm3_s(point.flow_cm3_s, flow.unit)
    flow_full_scale = abs(flow.value_max - flow.value_min)
    u_flow = flow.uncertainty.standard_uncertainty(flow_in_unit, flow_full_scale)
    flow_relative = (
        math.hypot(u_flow / abs(flow_in_unit), daq_relative) if flow_in_unit else math.inf
    )
    components.append(
        _component(
            name="flow rate",
            symbol="Q",
            evaluation_type="B",
            value=point.flow_cm3_s,
            unit="cm3/s",
            relative_uncertainty=flow_relative,
            sensitivity=1.0,
            dof=flow.uncertainty.dof,
            source=(
                f"{flow_name}: {flow.uncertainty.source}"
                if flow.uncertainty.source
                else f"{flow_name} specification"
            ),
        )
    )

    # -- inlet pressure ---------------------------------------------------
    u_p1_rel, p1_dof = _pressure_relative_uncertainty(
        calibration.inlet, point.inlet_pressure_atm, run, daq_relative
    )
    components.append(
        _component(
            name="inlet pressure",
            symbol="P1",
            evaluation_type="B",
            value=point.inlet_pressure_atm,
            unit="atm",
            relative_uncertainty=u_p1_rel,
            sensitivity=c_p1,
            dof=p1_dof,
            source=calibration.inlet.uncertainty.source or "inlet transducer",
        )
    )

    # -- downstream pressure (P2) -----------------------------------------
    if supplied_downstream:
        spec = run.downstream_pressure_uncertainty
        u_supplied_atm = units.to_atm(
            spec.standard_uncertainty(
                run.downstream_pressure, abs(run.downstream_pressure)
            ),
            run.downstream_pressure_unit,
        )
        u_p2_rel = u_supplied_atm / abs(point.downstream_pressure_atm)
        p2_dof = spec.dof
        p2_name = "downstream pressure"
        p2_source = spec.source or "operator-supplied value"
        notes.append(
            "P2 was supplied rather than measured, so its uncertainty is that of the "
            "stated value and it shares no transducer error with P1 -- the P1/P2 "
            "covariance term does not apply."
        )
    else:
        u_p2_rel, p2_dof = _pressure_relative_uncertainty(
            calibration.outlet, point.downstream_pressure_atm, run, daq_relative
        )
        p2_name = "outlet pressure"
        p2_source = calibration.outlet.uncertainty.source or "outlet transducer"

    components.append(
        _component(
            name=p2_name,
            symbol="P2",
            evaluation_type="B",
            value=point.downstream_pressure_atm,
            unit="atm",
            relative_uncertainty=u_p2_rel,
            sensitivity=c_p2,
            dof=p2_dof,
            source=p2_source,
        )
    )

    # -- reference pressure the flow is paired with -----------------------
    if flow.reading_basis == "standard":
        # A defined standard state, not a measurement.
        notes.append(
            "The flow reference pressure is the meter's defined standard state, so it "
            "contributes no measurement uncertainty."
        )
    else:
        at_supplied_outlet = supplied_downstream and flow.actual_pressure_source == "outlet"
        if at_supplied_outlet:
            # The meter sits on a line whose pressure was stated, not measured,
            # so charging it the outlet transducer's specification would bill a
            # calibration error to a number no transducer produced.
            u_ref_rel = u_p2_rel
            ref_dof = p2_dof
            ref_source = "operator-supplied downstream pressure"
            notes.append(
                "The flowmeter sits at the outlet, whose pressure was supplied rather "
                "than measured, so P_ref is that same stated value. It is perfectly "
                "correlated with P2; the budget treats them as independent, which "
                "errs high."
            )
        else:
            source_channel = (
                calibration.inlet
                if flow.actual_pressure_source == "inlet"
                else calibration.outlet
            )
            u_ref_rel, ref_dof = _pressure_relative_uncertainty(
                source_channel, point.reference_pressure_atm, run, daq_relative
            )
            ref_source = f"{flow.actual_pressure_source} transducer"
            notes.append(
                "flowmeter.reading_basis is 'actual', so the flow reference pressure is "
                "a transducer reading and is correlated with that channel; the budget "
                "treats it as independent, which errs high."
            )
        components.append(
            _component(
                name="flow reference pressure",
                symbol="P_ref",
                evaluation_type="B",
                value=point.reference_pressure_atm,
                unit="atm",
                relative_uncertainty=u_ref_rel,
                sensitivity=1.0,
                dof=ref_dof,
                source=ref_source,
            )
        )

    # -- viscosity model --------------------------------------------------
    components.append(
        _component(
            name="viscosity model",
            symbol="mu",
            evaluation_type="B",
            value=point.viscosity_cp,
            unit="cP",
            relative_uncertainty=run.gas.viscosity_relative_uncertainty,
            sensitivity=1.0,
            dof=math.inf,
            source=f"{run.gas.properties_source} transport correlation",
        )
    )

    # -- temperature, through the viscosity -------------------------------
    if viscosity_temperature_exponent:
        temperature_k = units.celsius_to_kelvin(point.temperature_c)
        u_temperature = hardware.temperature.uncertainty.standard_uncertainty(
            point.temperature_c, abs(point.temperature_c) or 1.0
        )
        components.append(
            _component(
                name="temperature",
                symbol="T",
                evaluation_type="B",
                value=temperature_k,
                unit="K",
                relative_uncertainty=u_temperature / temperature_k,
                sensitivity=viscosity_temperature_exponent,
                dof=hardware.temperature.uncertainty.dof,
                source="probe specification, via d(ln mu)/d(ln T)",
            )
        )

    # -- geometry ---------------------------------------------------------
    components.append(
        _component(
            name="sample length",
            symbol="L",
            evaluation_type="B",
            value=geometry.length_cm,
            unit="cm",
            relative_uncertainty=geometry.relative_length_uncertainty,
            sensitivity=1.0,
            dof=math.inf,
            source="caliper",
        )
    )
    components.append(
        _component(
            name="sample diameter",
            symbol="d",
            evaluation_type="B",
            value=geometry.diameter_cm,
            unit="cm",
            relative_uncertainty=geometry.diameter_uncertainty_cm / geometry.diameter_cm,
            sensitivity=-2.0,
            dof=math.inf,
            source="caliper, area goes as d^2",
        )
    )

    # -- bench repeatability ----------------------------------------------
    if hardware.uncertainty.repeatability_relative > 0.0:
        components.append(
            _component(
                name="bench repeatability",
                symbol="rep",
                evaluation_type="B",
                value=point.permeability_darcy,
                unit="D",
                relative_uncertainty=hardware.uncertainty.repeatability_relative,
                sensitivity=1.0,
                dof=math.inf,
                source=hardware.uncertainty.notes or "estimated from repeat loadings",
            )
        )

    # -- Type A: the run's own scatter ------------------------------------
    if type_a_relative is not None and run.uncertainty.include_type_a:
        components.append(
            _component(
                name="steady-state scatter",
                symbol="s/sqrt(n)",
                evaluation_type="A",
                value=point.permeability_darcy,
                unit="D",
                relative_uncertainty=type_a_relative,
                sensitivity=1.0,
                dof=type_a_dof,
                source="standard deviation of the mean over the steady-state window",
            )
        )

    # -- combine ----------------------------------------------------------
    # P1 and P2 enter with opposite signs, so a shared transducer error partly
    # cancels. It does not apply to a supplied downstream pressure, which is a
    # stated constant rather than a reading and shares nothing with P1.
    correlations = (
        [
            _Correlation(
                label="P1/P2",
                sensitivity_a=c_p1,
                sensitivity_b=c_p2,
                relative_uncertainty_a=u_p1_rel,
                relative_uncertainty_b=u_p2_rel,
                coefficient=calibration.correlation,
                explanation="because the two pressures enter with opposite signs",
            )
        ]
        if not supplied_downstream
        else []
    )

    return combine_budget(
        components,
        value_darcy=point.permeability_darcy,
        report=run.uncertainty,
        correlations=correlations,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Pulse decay
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PulseDecayPoint:
    """The fitted decay and the state it was measured at, in internal units."""

    permeability_darcy: float
    decay_rate_per_s: float
    mean_pressure_atm: float
    viscosity_cp: float
    gas_compressibility_per_atm: float
    temperature_c: float
    upstream_volume_cm3: float
    downstream_volume_cm3: float
    #: The spacers that made up ``upstream_volume_cm3``. Needed here because
    #: their uncertainty depends on which bores and how many of each, not just
    #: on the total volume they added.
    upstream_spacers: tuple = ()
    #: ``None`` when the zero-storage (Brace) form was used, in which case
    #: porosity is not an input to the measurement at all.
    porosity_fraction: float | None = None
    #: ``theta_1``, present only under the storage correction.
    storage_root: float | None = None


def _log_sensitivity(
    evaluate, name: str, value: float, *, delta_frac: float = 1.0e-4
) -> float:
    """``d ln k / d ln x`` by central difference.

    Under the Dicker-Smits correction the exponents of L, d, phi, V1 and V2 are
    no longer constants -- they depend on the storage ratios, converging to the
    Brace values only as those go to zero. Differencing the actual model is
    exact at the operating point and self-maintaining, and it correctly reports
    about zero for porosity when the correction is negligible, which is more
    informative than declaring the term ignored.
    """
    if value <= 0.0:
        return 0.0
    low = evaluate({name: value * (1.0 - delta_frac)})
    high = evaluate({name: value * (1.0 + delta_frac)})
    if not low or not high or low <= 0.0 or high <= 0.0:
        return 0.0
    return math.log(high / low) / math.log((1.0 + delta_frac) / (1.0 - delta_frac))


def _pulse_decay_sensitivities(
    point: PulseDecayPoint, geometry: SampleGeometry
) -> dict[str, float]:
    """Relative sensitivity of k to each geometric and volumetric input.

    ``alpha``, ``mu`` and ``c_g`` factor out of the storage root entirely, so
    their exponent is exactly +1 in both models and they are not differenced
    here. Everything else is.
    """
    from gasperm import pulse_decay as physics

    storage = point.porosity_fraction is not None and point.storage_root is not None
    if not storage:
        # Brace: exact, and written as literals so the closed form stays visible
        # rather than having to be inferred from a difference.
        total = point.upstream_volume_cm3 + point.downstream_volume_cm3
        return {
            "length_cm": 1.0,
            "diameter_cm": -2.0,
            "porosity_fraction": 0.0,
            "upstream_volume_cm3": point.downstream_volume_cm3 / total,
            "downstream_volume_cm3": point.upstream_volume_cm3 / total,
        }

    base = dict(
        decay_rate_per_s=point.decay_rate_per_s,
        viscosity_cp=point.viscosity_cp,
        gas_compressibility_per_atm=point.gas_compressibility_per_atm,
        length_cm=geometry.length_cm,
        area_cm2=geometry.area_cm2,
        porosity_fraction=point.porosity_fraction,
        upstream_volume_cm3=point.upstream_volume_cm3,
        downstream_volume_cm3=point.downstream_volume_cm3,
    )

    def evaluate(overrides: dict[str, float]) -> float | None:
        args = dict(base)
        # Diameter is not an argument of the model; area is. Convert here so the
        # caller can difference the quantity the caliper actually measures.
        if "diameter_cm" in overrides:
            args["area_cm2"] = units.circle_area_cm2(overrides["diameter_cm"])
        else:
            args.update(overrides)
        try:
            return physics.dicker_smits_permeability_darcy(**args)
        except physics.PulseDecayInputError:
            return None

    return {
        "length_cm": _log_sensitivity(evaluate, "length_cm", geometry.length_cm),
        "diameter_cm": _log_sensitivity(evaluate, "diameter_cm", geometry.diameter_cm),
        "porosity_fraction": _log_sensitivity(
            evaluate, "porosity_fraction", point.porosity_fraction
        ),
        "upstream_volume_cm3": _log_sensitivity(
            evaluate, "upstream_volume_cm3", point.upstream_volume_cm3
        ),
        "downstream_volume_cm3": _log_sensitivity(
            evaluate, "downstream_volume_cm3", point.downstream_volume_cm3
        ),
    }


def build_pulse_decay_budget(
    point: PulseDecayPoint,
    geometry: SampleGeometry,
    hardware: HardwareConfig,
    run: RunConfig,
    *,
    decay_rate_relative_uncertainty: float,
    decay_rate_dof: float = math.inf,
    viscosity_temperature_exponent: float = 0.0,
    compressibility_pressure_exponent: float = -1.0,
    porosity_uncertainty: float | None = None,
) -> UncertaintyBudget:
    """GUM budget for ``k = alpha mu c_g L / (A (1/V1 + 1/V2))``.

    **The transducers are largely absent, and that is the point.** ``alpha`` is
    a *rate*: scaling the differential by a constant gain leaves it unchanged,
    and the fitted offset absorbs a constant zero error exactly. So transducer
    gain and zero errors -- the bulk of a datasheet accuracy figure -- cancel
    out of this measurement. What does not cancel is their noise and their
    nonlinearity over the pulse range, and those show up empirically as the
    scatter of the fit, i.e. as the Type A ``u(alpha)`` below. The transducers
    re-enter only through the mean pore pressure, which sets ``c_g``.

    Args:
        point: The fitted decay and the state it was measured at.
        geometry: Plug geometry with its caliper uncertainties.
        hardware: Vessel volumes and instrument specifications.
        run: Coverage settings and the gas model's uncertainties.
        decay_rate_relative_uncertainty: ``u(alpha)/alpha`` from the fit.
        decay_rate_dof: Degrees of freedom of that fit.
        viscosity_temperature_exponent: ``d ln mu / d ln T``.
        compressibility_pressure_exponent: ``d ln c / d ln P``, about -1.
        porosity_uncertainty: ``u(phi)``. Only used under the storage
            correction, where porosity is an input rather than metadata.
    """
    notes: list[str] = [
        "Pulse decay measures a RATE, so a constant gain or zero error in the "
        "differential cancels: scaling dP leaves alpha unchanged and the fitted "
        "offset absorbs a constant. The transducers therefore enter only through "
        "the mean pore pressure (via the gas compressibility) and through their "
        "noise, which u(alpha) below already captures empirically."
    ]
    components: list[UncertaintyComponent] = []
    sensitivities = _pulse_decay_sensitivities(point, geometry)

    # -- Type A: the decay fit itself -------------------------------------
    # Included unconditionally. run.uncertainty.include_type_a does not gate it:
    # this is not optional scatter, it is the measurement's own precision, and a
    # budget without it would say nothing about how well the decay was resolved.
    components.append(
        _component(
            name="decay rate fit",
            symbol="alpha",
            evaluation_type="A",
            value=point.decay_rate_per_s,
            unit="1/s",
            relative_uncertainty=decay_rate_relative_uncertainty,
            sensitivity=1.0,
            dof=decay_rate_dof,
            source="standard error of the fitted exponential decay rate",
        )
    )
    if not run.uncertainty.include_type_a:
        notes.append(
            "run.uncertainty.include_type_a is false, but the decay-rate term is "
            "included anyway: it is the pulse-decay measurement itself, not a "
            "scatter term that can be left out."
        )

    # -- gas model --------------------------------------------------------
    components.append(
        _component(
            name="viscosity model",
            symbol="mu",
            evaluation_type="B",
            value=point.viscosity_cp,
            unit="cP",
            relative_uncertainty=run.gas.viscosity_relative_uncertainty,
            sensitivity=1.0,
            dof=math.inf,
            source=f"{run.gas.properties_source} viscosity for {run.gas.name}",
        )
    )
    components.append(
        _component(
            name="gas compressibility",
            symbol="c_g",
            evaluation_type="B",
            value=point.gas_compressibility_per_atm,
            unit="1/atm",
            relative_uncertainty=run.gas.compressibility_relative_uncertainty,
            sensitivity=1.0,
            dof=math.inf,
            source=f"{run.gas.properties_source} compressibility for {run.gas.name}",
        )
    )

    # -- mean pore pressure, through the compressibility -------------------
    u_pressure_rel, pressure_dof = _pressure_relative_uncertainty(
        hardware.pressure_calibration.inlet,
        point.mean_pressure_atm,
        run,
        hardware.uncertainty.daq_relative,
    )
    components.append(
        _component(
            name="mean pore pressure",
            symbol="P_mean",
            evaluation_type="B",
            value=point.mean_pressure_atm,
            unit="atm",
            relative_uncertainty=u_pressure_rel,
            sensitivity=compressibility_pressure_exponent,
            dof=pressure_dof,
            source="transducer specification, via d(ln c)/d(ln P)",
        )
    )

    # -- temperature, through the viscosity -------------------------------
    if viscosity_temperature_exponent:
        temperature_k = units.celsius_to_kelvin(point.temperature_c)
        u_temperature = hardware.temperature.uncertainty.standard_uncertainty(
            point.temperature_c, abs(point.temperature_c) or 1.0
        )
        components.append(
            _component(
                name="temperature",
                symbol="T",
                evaluation_type="B",
                value=temperature_k,
                unit="K",
                relative_uncertainty=u_temperature / temperature_k,
                sensitivity=viscosity_temperature_exponent,
                dof=hardware.temperature.uncertainty.dof,
                source="probe specification, via d(ln mu)/d(ln T)",
            )
        )

    # -- geometry ---------------------------------------------------------
    components.append(
        _component(
            name="sample length",
            symbol="L",
            evaluation_type="B",
            value=geometry.length_cm,
            unit="cm",
            relative_uncertainty=geometry.relative_length_uncertainty,
            sensitivity=sensitivities["length_cm"],
            dof=math.inf,
            source="caliper",
        )
    )
    components.append(
        _component(
            name="sample diameter",
            symbol="d",
            evaluation_type="B",
            value=geometry.diameter_cm,
            unit="cm",
            relative_uncertainty=(
                geometry.diameter_uncertainty_cm / geometry.diameter_cm
                if geometry.diameter_cm
                else math.inf
            ),
            sensitivity=sensitivities["diameter_cm"],
            dof=math.inf,
            source="caliper",
        )
    )

    # -- the volumes ------------------------------------------------------
    # A vessel has no meaningful "full scale", so each spec is evaluated with
    # the reading passed as both -- which makes percent_full_scale and
    # percent_reading coincide rather than quietly meaning something odd. The
    # composition of each side (vessel + dead volume + any spacer stack) and
    # the accumulation of its uncertainty belong to the config, which is what
    # knows how the side is built; this only consumes the totals.
    reservoirs = hardware.reservoirs
    spacers = point.upstream_spacers
    volume_relatives: list[float] = []
    for symbol, name, vessel, volume_cm3, u_volume, key in (
        (
            "V1",
            "upstream volume",
            reservoirs.upstream,
            point.upstream_volume_cm3,
            reservoirs.upstream_uncertainty_cm3(spacers),
            "upstream_volume_cm3",
        ),
        (
            "V2",
            "downstream volume",
            reservoirs.downstream,
            point.downstream_volume_cm3,
            reservoirs.downstream_uncertainty_cm3(),
            "downstream_volume_cm3",
        ),
    ):
        relative = u_volume / volume_cm3 if volume_cm3 else math.inf
        volume_relatives.append(relative)
        components.append(
            _component(
                name=name,
                symbol=symbol,
                evaluation_type="B",
                value=volume_cm3,
                unit="cm3",
                relative_uncertainty=relative,
                sensitivity=sensitivities[key],
                dof=vessel.uncertainty.dof,
                source=vessel.method or vessel.uncertainty.source or "vessel calibration",
            )
        )
    notes.append(
        "The volumes are DEAD volumes -- vessel plus tubing, ports and valve "
        "internals up to the plug face, plus any upstream spacers. Permeability is "
        "directly proportional to them, so anything left out of the figure is a "
        "systematic error here."
    )
    if spacers:
        added = reservoirs.spacer_volume_cm3(spacers)
        stack = ", ".join(str(fitting) for fitting in spacers)
        notes.append(
            f"{len(spacers)} upstream spacer{'s' if len(spacers) != 1 else ''} "
            f"[{stack}] add {added:.4g} cm3 to V1 "
            f"({added / point.upstream_volume_cm3:.1%} of it). Their bore error is "
            "shared within a type, so it sums; their lengths are measured "
            "separately, so those add in quadrature."
        )

    # -- porosity, only when the storage correction is in use --------------
    if point.porosity_fraction is not None:
        if porosity_uncertainty:
            components.append(
                _component(
                    name="porosity",
                    symbol="phi",
                    evaluation_type="B",
                    value=point.porosity_fraction,
                    unit="",
                    relative_uncertainty=porosity_uncertainty / point.porosity_fraction,
                    sensitivity=sensitivities["porosity_fraction"],
                    dof=math.inf,
                    source="sample.porosity_uncertainty",
                )
            )
        else:
            notes.append(
                "The Dicker-Smits storage correction is in use, so porosity is an "
                "input to this measurement, but sample.porosity_uncertainty is "
                "unrecorded and its term is omitted. Its sensitivity here is "
                f"{sensitivities['porosity_fraction']:+.3f}."
            )
    else:
        notes.append(
            "The zero-storage (Brace) form was used, so porosity does not enter the "
            "result at all."
        )

    # -- bench repeatability ----------------------------------------------
    if hardware.uncertainty.repeatability_relative > 0.0:
        components.append(
            _component(
                name="bench repeatability",
                symbol="rep",
                evaluation_type="B",
                value=point.permeability_darcy,
                unit="D",
                relative_uncertainty=hardware.uncertainty.repeatability_relative,
                sensitivity=1.0,
                dof=math.inf,
                source=hardware.uncertainty.notes or "estimated from repeat pulses",
            )
        )

    correlations = [
        _Correlation(
            label="V1/V2",
            sensitivity_a=sensitivities["upstream_volume_cm3"],
            sensitivity_b=sensitivities["downstream_volume_cm3"],
            relative_uncertainty_a=volume_relatives[0],
            relative_uncertainty_b=volume_relatives[1],
            coefficient=reservoirs.correlation,
            explanation="because both volumes enter with the same sign, unlike P1/P2,",
        )
    ]

    return combine_budget(
        components,
        value_darcy=point.permeability_darcy,
        report=run.uncertainty,
        correlations=correlations,
        measurand="apparent gas permeability (pulse decay)",
        notes=notes,
    )
