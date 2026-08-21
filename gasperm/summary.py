"""Everything one core plug has been through, assembled into one report.

A plug accumulates history: several pressure steps for a Klinkenberg series, a
leak test or two, an aborted run, a re-derivation after a calibration was
corrected, and -- for an exposure study -- the whole lot again months later.
That history lives as a directory of timestamped runs, which is the right way
to *store* it and a poor way to *read* it.

This module reduces it to one answer per plug. It is hardware-free and does no
physics of its own: every number comes from the runs' own stored summaries, or
from :mod:`gasperm.klinkenberg` regressing the points they reduce to.

Two things it does that a listing cannot:

**It reports the gaps, not just the contents.** A summary whose only job was to
restate what is on disk would leave the operator to notice that one run never
confirmed, that two were measured on different meters, or that a pulse-decay
series has no leak test behind it. Those are the findings; the table is the
evidence.

**It notices when the history is two campaigns rather than one.** Runs cluster
in time -- a day of pressure steps, then a month of nothing, then another day.
When that gap is unmistakable the summary names the date and suggests
``compare --split``, because a plug measured twice either side of a treatment is
a paired experiment whose result is the *difference*, and reading two separate
Klinkenberg fits off one page is not the same thing.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from gasperm.models import KlinkenbergResult, RunSummary

logger = logging.getLogger(__name__)

__all__ = [
    "PlugHistory",
    "RunLine",
    "SampleReport",
    "build_report",
    "detect_campaign_split",
]

#: A quiet stretch has to be both **relatively** and **absolutely** long before
#: it is called a break between campaigns: several times the plug's own typical
#: spacing, and at least this. Pressure steps hours apart must never look like
#: two campaigns, and a genuine exposure lasts days or weeks.
_MIN_CAMPAIGN_GAP = timedelta(days=3)

#: How many times the median spacing a gap must exceed to count.
_GAP_RATIO = 5.0


@dataclass(frozen=True)
class RunLine:
    """One run, flattened to the fields a summary table shows."""

    name: str
    started_at: datetime | None
    method: str
    purpose: str
    mean_pressure_atm: float | None
    permeability_darcy: float | None
    expanded_uncertainty_darcy: float | None
    confirmed: bool
    flowmeter: str | None
    downstream_convention: str | None
    gas_name: str | None = None
    temperature_c: float | None = None
    duration_s: float | None = None
    derived_from: str | None = None
    #: The pair behind ``mean_pressure_atm``. ``None`` for a run recorded
    #: before the summary carried them -- they cannot be recovered from a mean,
    #: so they read as unknown rather than being split evenly and guessed at.
    inlet_pressure_atm: float | None = None
    downstream_pressure_atm: float | None = None
    #: ``dP0``, the pulse a decay started from. ``None`` for steady state,
    #: where there is no pulse, and for a pulse run that never fitted one.
    pulse_amplitude_atm: float | None = None
    #: The vessel pressures **at the pulse**: a pulse-decay run's setup
    #: condition. ``None`` for steady state, and for pulse runs recorded before
    #: these were kept.
    initial_inlet_pressure_atm: float | None = None
    initial_downstream_pressure_atm: float | None = None
    #: Why this run is not a usable measurement, when it is not.
    excluded_reason: str = ""

    @property
    def relative_uncertainty(self) -> float | None:
        if not self.permeability_darcy or self.expanded_uncertainty_darcy is None:
            return None
        return self.expanded_uncertainty_darcy / abs(self.permeability_darcy)

    @property
    def reported_inlet_pressure_atm(self) -> float | None:
        """The inlet pressure that characterises this run.

        For steady state, the mean over the measured window -- the pressure the
        equation used. For pulse decay, the pressure **at the pulse**: the
        upstream vessel decays toward the downstream for the whole run, so its
        mean is very nearly the pore pressure and describes nothing that could
        be set up again. What an operator re-measuring the plug needs is the
        charge pressure, and that is what this returns.

        Falls back to the mean for a pulse run recorded before the setup
        condition was kept, which is the only number that run has.
        """
        if self.method == "pulse_decay" and self.initial_inlet_pressure_atm is not None:
            return self.initial_inlet_pressure_atm
        return self.inlet_pressure_atm

    @property
    def reported_downstream_pressure_atm(self) -> float | None:
        """The outlet pressure, chosen to match :attr:`reported_inlet_pressure_atm`.

        Taken at the same instant as the inlet, so on a pulse-decay row the two
        columns and ``dP0`` describe one moment and ``dP0`` is their difference.
        Mixing an initial inlet with a mean outlet would make that subtraction
        wrong by whatever the decay had already done.
        """
        if (
            self.method == "pulse_decay"
            and self.initial_downstream_pressure_atm is not None
        ):
            return self.initial_downstream_pressure_atm
        return self.downstream_pressure_atm


@dataclass(frozen=True)
class PlugHistory:
    """The raw material: what discovery found for one plug."""

    sample_id: str
    summaries: tuple[RunSummary, ...]
    lines: tuple[RunLine, ...]
    leak_tests: tuple[RunLine, ...]
    excluded: tuple[RunLine, ...]


@dataclass(frozen=True)
class SampleReport:
    """One plug's whole history, reduced to what an operator needs to read."""

    sample_id: str
    #: Descriptive fields, taken from the most recent run that recorded them.
    description: str = ""
    lithology: str = ""
    formation: str = ""
    well: str = ""
    depth: float | None = None
    depth_unit: str = ""
    length_cm: float | None = None
    diameter_cm: float | None = None
    porosity_fraction: float | None = None
    porosity_method: str = ""

    measurements: tuple[RunLine, ...] = ()
    leak_tests: tuple[RunLine, ...] = ()
    excluded: tuple[RunLine, ...] = ()
    klinkenberg: KlinkenbergResult | None = None

    methods: tuple[str, ...] = ()
    gases: tuple[str, ...] = ()
    flowmeters: tuple[str, ...] = ()
    conventions: tuple[str, ...] = ()
    first_run: datetime | None = None
    last_run: datetime | None = None
    #: The instant a plausible break between campaigns falls on, if there is one.
    campaign_split: datetime | None = None

    findings: tuple[str, ...] = ()

    @property
    def run_count(self) -> int:
        return len(self.measurements)

    @property
    def pressure_range_atm(self) -> tuple[float, float] | None:
        pressures = [
            line.mean_pressure_atm
            for line in self.measurements
            if line.mean_pressure_atm is not None
        ]
        if not pressures:
            return None
        return (min(pressures), max(pressures))

    @property
    def distinct_pressures(self) -> int:
        return len(
            {
                round(line.mean_pressure_atm, 6)
                for line in self.measurements
                if line.mean_pressure_atm is not None
            }
        )


def _line_from(record, summary: RunSummary | None) -> RunLine:
    """One table row, preferring the stored summary and falling back to discovery."""
    confirmed = bool(
        summary.measurement_confirmed
        if summary is not None
        else record.measurement_confirmed
    )
    return RunLine(
        name=record.name,
        started_at=record.started_at,
        method=(summary.method if summary else record.method) or "steady_state",
        purpose=(summary.purpose if summary else record.purpose) or "measurement",
        mean_pressure_atm=(
            summary.mean_pressure_atm if summary else record.mean_pressure_atm
        ),
        permeability_darcy=(
            summary.permeability_darcy if summary else record.permeability_darcy
        ),
        expanded_uncertainty_darcy=(
            summary.uncertainty.expanded_uncertainty_darcy
            if summary is not None and summary.uncertainty is not None
            else None
        ),
        confirmed=confirmed,
        flowmeter=record.flowmeter,
        downstream_convention=record.downstream_convention,
        gas_name=summary.gas_name if summary else None,
        temperature_c=summary.mean_temperature_c if summary else None,
        duration_s=summary.duration_s if summary else None,
        derived_from=record.derived_from,
        inlet_pressure_atm=summary.mean_inlet_pressure_atm if summary else None,
        downstream_pressure_atm=(
            summary.mean_downstream_pressure_atm if summary else None
        ),
        pulse_amplitude_atm=(
            summary.pulse_decay.pulse_amplitude_atm
            if summary is not None and summary.pulse_decay is not None
            else None
        ),
        initial_inlet_pressure_atm=(
            summary.pulse_decay.initial_upstream_pressure_atm
            if summary is not None and summary.pulse_decay is not None
            else None
        ),
        initial_downstream_pressure_atm=(
            summary.pulse_decay.initial_downstream_pressure_atm
            if summary is not None and summary.pulse_decay is not None
            else None
        ),
        excluded_reason=(
            ""
            if confirmed
            else "never confirmed a measurement"
        ),
    )


def detect_campaign_split(moments: Sequence[datetime]) -> datetime | None:
    """The instant a plug's history plausibly divides into two campaigns.

    Returns the start of the run that opens the later campaign, which is the
    value ``compare --split`` wants. ``None`` when the runs are one continuous
    stretch -- which is the ordinary case, and must stay silent.
    """
    stamps = sorted(m for m in moments if m is not None)
    if len(stamps) < 4:
        # Two runs either side is the fewest that can be called two campaigns;
        # below that a "gap" is just the spacing.
        return None
    gaps = [
        (later - earlier, later) for earlier, later in zip(stamps, stamps[1:])
    ]
    median = statistics.median(gap for gap, _ in gaps)
    widest, opens_at = max(gaps, key=lambda item: item[0])
    if widest < _MIN_CAMPAIGN_GAP:
        return None
    if median.total_seconds() > 0 and widest < median * _GAP_RATIO:
        return None
    # Both sides need enough runs to stand on their own.
    before = sum(1 for stamp in stamps if stamp < opens_at)
    if before < 2 or len(stamps) - before < 2:
        return None
    return opens_at


def _findings(report: SampleReport) -> list[str]:
    """What is worth saying about this plug beyond the table.

    Ordered by how much it should change what the operator does next.
    """
    findings: list[str] = []

    # First, because it changes which command to run rather than merely
    # annotating this one -- and because a fit regressed across two states of
    # the same plug is what a poor R^2 above is usually telling you.
    if report.campaign_split is not None:
        stamp = report.campaign_split.date().isoformat()
        findings.append(
            f"These runs fall into two groups either side of {stamp}. If something "
            "happened to the plug in between, they are not one series: the measurand "
            "is the *change*, and any fit spanning both is regressing two states as "
            f"one. 'gasperm compare {report.sample_id} --split {stamp}' reports it "
            "with the errors common to both campaigns cancelled out."
        )

    if report.klinkenberg is None:
        if report.distinct_pressures < 2:
            findings.append(
                f"No Klinkenberg fit: {report.run_count} confirmed run(s) at "
                f"{report.distinct_pressures} distinct mean pressure(s). The "
                "correction regresses k_g against 1/P_mean, so it needs at least two."
            )
        else:
            findings.append(
                "No Klinkenberg fit, although there is pressure spread -- run "
                "'gasperm klinkenberg --sample "
                f"{report.sample_id}' to see why it was refused."
            )
    elif report.klinkenberg.point_count < 3:
        findings.append(
            "The Klinkenberg fit rests on 2 points, which define a line exactly and "
            "leave nothing over to check it with. A third pressure is what turns the "
            "fit into a measurement."
        )

    fit = report.klinkenberg
    if fit is not None and fit.liquid_permeability_darcy <= 0.0:
        findings.append(
            f"k_L is {fit.liquid_permeability_darcy:.3g} D, i.e. not positive. A "
            "negative intercept means the apparent permeabilities fall the wrong way "
            "with pressure -- usually a flowmeter reporting its own zero offset, "
            "which is what pulse decay exists to avoid."
        )
    if fit is not None and fit.slippage_factor_atm < 0.0:
        findings.append(
            f"The slippage factor b is negative ({fit.slippage_factor_atm:.3g} atm). "
            "Gas slip raises apparent permeability at low pressure, never lowers it."
        )

    if len(report.methods) > 1:
        findings.append(
            "This plug was measured by more than one method ("
            + ", ".join(report.methods)
            + "). They carry a systematic offset relative to each other, so "
            "klinkenberg refuses to mix them without --allow-mixed-methods."
        )
    if len(report.flowmeters) > 1:
        findings.append(
            "More than one flowmeter was used ("
            + ", ".join(report.flowmeters)
            + "). Each has its own calibration error, so it does not cancel between "
            "these runs."
        )
    if len(report.conventions) > 1:
        findings.append(
            "The downstream pressure was obtained differently across these runs ("
            + ", ".join(report.conventions)
            + "). P2 sets the mean pressure the regression plots against."
        )
    if len(report.gases) > 1:
        findings.append(
            "More than one gas was used (" + ", ".join(report.gases) + "). Viscosity "
            "and slippage both depend on it."
        )

    if "pulse_decay" in report.methods and not report.leak_tests:
        findings.append(
            "No leak test is recorded for this plug's rig. On a pulse-decay run a "
            "leak decays the differential exactly as the sample does; without one "
            "there is no floor below which a result means nothing. Run "
            "'gasperm collect --leak-test'."
        )

    for line in report.excluded:
        findings.append(f"{line.name} is not a measurement: {line.excluded_reason}.")

    derived = [line for line in report.measurements if line.derived_from]
    for line in derived:
        findings.append(
            f"{line.name} was re-derived from {line.derived_from}, which it "
            "supersedes; the original is on disk but is not counted here."
        )
    return findings


def build_report(
    sample_id: str,
    records: Sequence,
    summaries: Sequence[RunSummary | None],
    *,
    klinkenberg: KlinkenbergResult | None = None,
) -> SampleReport:
    """Assemble one plug's report from its discovered runs.

    Args:
        sample_id: The plug.
        records: Its ``RunRecord``s, already de-superseded by the caller.
        summaries: The stored ``RunSummary`` for each record, positionally, with
            ``None`` where a run produced none.
        klinkenberg: The fit over its usable points, when one could be made.
    """
    lines = [_line_from(record, summary) for record, summary in zip(records, summaries)]

    leak_tests = tuple(line for line in lines if line.purpose == "leak_test")
    candidates = [line for line in lines if line.purpose != "leak_test"]
    measurements = tuple(line for line in candidates if line.confirmed)
    excluded = tuple(line for line in candidates if not line.confirmed)

    # Descriptive fields come from the most recent run that recorded them: a
    # plug re-measured with better calipers should read as its latest geometry,
    # not its first.
    descriptive = [
        summary.metadata
        for summary in sorted(
            (s for s in summaries if s is not None and s.metadata is not None),
            key=lambda s: s.started_at,
        )
    ]
    latest = descriptive[-1] if descriptive else None

    report = SampleReport(
        sample_id=sample_id,
        description=getattr(latest, "sample_description", "") or "",
        lithology=getattr(latest, "lithology", "") or "",
        formation=getattr(latest, "formation", "") or "",
        well=getattr(latest, "well", "") or "",
        depth=getattr(latest, "depth", None),
        depth_unit=getattr(latest, "depth_unit", "") or "",
        length_cm=getattr(latest, "length_cm", None),
        diameter_cm=getattr(latest, "diameter_cm", None),
        porosity_fraction=getattr(latest, "porosity_fraction", None),
        porosity_method=getattr(latest, "porosity_method", "") or "",
        measurements=measurements,
        leak_tests=leak_tests,
        excluded=excluded,
        klinkenberg=klinkenberg,
        methods=tuple(sorted({line.method for line in measurements})),
        gases=tuple(sorted({line.gas_name for line in measurements if line.gas_name})),
        flowmeters=tuple(
            sorted({line.flowmeter for line in measurements if line.flowmeter})
        ),
        conventions=tuple(
            sorted(
                {
                    line.downstream_convention
                    for line in measurements
                    if line.downstream_convention
                }
            )
        ),
        first_run=min(
            (line.started_at for line in lines if line.started_at), default=None
        ),
        last_run=max(
            (line.started_at for line in lines if line.started_at), default=None
        ),
        campaign_split=detect_campaign_split(
            [line.started_at for line in measurements if line.started_at]
        ),
    )
    return _with_findings(report)


def _with_findings(report: SampleReport) -> SampleReport:
    from dataclasses import replace

    return replace(report, findings=tuple(_findings(report)))


def total_measured_time_s(report: SampleReport) -> float:
    """How long this plug has actually been on the rig, in seconds.

    Worth stating on a tight-rock study: a plug can represent days of bench
    time, and that is the number that decides whether another pressure step is
    affordable.
    """
    return math.fsum(
        line.duration_s or 0.0 for line in (*report.measurements, *report.leak_tests)
    )
