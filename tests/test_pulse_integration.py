"""Pulse decay wired end to end: DAQ, loop, budget, storage, Klinkenberg.

The physics is covered in ``test_pulse_decay.py``. What is checked here is that
the pieces are connected correctly -- that a run reads two channels and not
three, that a fitted decay survives a round trip through the sidecar, and that
the regression refuses to mix methods. Every synthetic decay is driven through
the real calibration, so nothing here is a tautology.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from gasperm.acquisition import (
    PulseDecayLoop,
    PulseProcessor,
    format_pulse_reading_line,
    pulse_console_header,
    summarize_pulse_decay_run,
)
from gasperm.config import GaspermConfig
from gasperm.hardware.daq import build_channel_specs
from gasperm.klinkenberg import fit_klinkenberg
from gasperm.models import KlinkenbergPoint
from gasperm.pulse_decay import dicker_smits_decay_rate_per_s
from gasperm.storage import (
    READING_COLUMNS,
    RunWriter,
    describe_method,
    point_from_run,
    read_readings_csv,
    run_method,
)

from conftest import decay_voltages


def true_decay_rate(config: GaspermConfig, permeability_darcy: float, mean_atm: float):
    """The rate this plug and these vessels would actually produce."""
    geometry = config.geometry()
    return dicker_smits_decay_rate_per_s(
        permeability_darcy=permeability_darcy,
        viscosity_cp=0.0178,
        gas_compressibility_per_atm=1.0 / mean_atm,
        length_cm=geometry.length_cm,
        area_cm2=geometry.area_cm2,
        porosity_fraction=config.sample.porosity_fraction,
        upstream_volume_cm3=config.hardware.reservoirs.upstream_volume_cm3(
            config.run.pulse_decay.upstream_spacers
        ),
        downstream_volume_cm3=config.hardware.reservoirs.downstream_volume_cm3(),
    )


class _Clock:
    """A clock tied to the sample index, not to how often it is asked.

    The loop reads its clock more than once per sample, so a per-call tick would
    make the recorded elapsed time run at twice the rate the synthesised decay
    does -- and the fit would come back at exactly half the true alpha.
    """

    def __init__(self, source, step_s: float) -> None:
        self._source = source
        self._step = step_s

    def __call__(self) -> float:
        return self._source.read_count * self._step


def run_decay(config, fake_analog_source, fake_temperature_source, *, rate, **kwargs):
    """Drive a full pulse-decay run over a synthesised decay."""
    step_s = kwargs.pop("step_s", 0.1)
    frames = decay_voltages(config, decay_rate_per_s=rate, step_s=step_s, **kwargs)
    source = fake_analog_source(frames)
    config.run.max_samples = len(frames)
    processor = PulseProcessor(config, _fixed_provider())
    loop = PulseDecayLoop(
        config,
        processor,
        source,
        fake_temperature_source(),
        clock=_Clock(source, step_s),
        sleep=lambda _: None,
    )
    loop.run(install_signal_handler=False)
    return loop


def _fixed_provider():
    from gasperm.gas_properties import FixedPropertyProvider

    return FixedPropertyProvider("Nitrogen", 0.0178, reason="test fixture")


class TestChannelSelection:
    def test_pulse_mode_opens_two_channels_and_no_flowmeter(self, pulse_config):
        """Requirement: pulse decay reads no flow, so it opens no flow input."""
        specs = build_channel_specs(pulse_config)
        assert len(specs) == 2
        assert [spec.role for spec in specs] == ["inlet pressure", "outlet pressure"]

    def test_steady_state_still_opens_three(self, base_config):
        specs = build_channel_specs(base_config)
        assert len(specs) == 3
        assert specs[-1].role == "flow"

    def test_a_dedicated_pulse_pair_is_used_when_present(self, pulse_config):
        from gasperm.config import PulseTransducersConfig

        pulse_config.hardware.pulse_transducers = PulseTransducersConfig()
        specs = build_channel_specs(pulse_config)
        assert [spec.name for spec in specs] == ["ai4", "ai5"]
        # 0-10 V, unlike the 0-5 V steady-state pair: per-channel ranges matter.
        assert specs[0].max_volts == pytest.approx(10.0)

    def test_the_pulse_pair_may_not_collide_with_the_steady_pair(self, base_config):
        from gasperm.config import HardwareConfig, PulseTransducerConfig
        from gasperm.config.hardware import PulseTransducersConfig

        with pytest.raises(ValueError, match="already assigned"):
            HardwareConfig(
                pulse_transducers=PulseTransducersConfig(
                    upstream=PulseTransducerConfig(channel="ai0")
                )
            )


class TestRunEndToEnd:
    def test_a_synthesised_decay_recovers_its_permeability(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """The whole path: volts -> calibration -> monitor -> fit -> k."""
        k_true = 5.0e-4
        rate = true_decay_rate(pulse_config, k_true, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        summary = loop.summarize()
        assert summary.method == "pulse_decay"
        assert summary.measurement_confirmed
        assert summary.permeability_darcy == pytest.approx(k_true, rel=0.01)

    def test_a_transducer_zero_mismatch_is_fitted_out(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """The reason the fit is nonlinear, exercised through the real loop."""
        k_true = 5.0e-4
        rate = true_decay_rate(pulse_config, k_true, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0, offset_atm=0.002,
        )
        summary = loop.summarize()
        assert summary.pulse_decay.fitted_offset_atm == pytest.approx(0.002, rel=0.05)
        assert summary.permeability_darcy == pytest.approx(k_true, rel=0.02)

    def test_the_setup_condition_is_captured_at_the_pulse(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """What the rig has to be charged to in order to repeat this run.

        The synthesised decay holds the vessels at ``P +/- dP/2``, so at the
        pulse they are 10.25 and 9.75 atm. Recording those is the whole point:
        an operator re-measuring the plug sets a regulator, and no *mean* over
        the run is a number a regulator can be set to.
        """
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        result = loop.summarize().pulse_decay
        assert result.initial_upstream_pressure_atm == pytest.approx(10.25, rel=1e-3)
        assert result.initial_downstream_pressure_atm == pytest.approx(9.75, rel=1e-3)

    def test_the_setup_condition_is_not_the_run_mean(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """The distinction that makes this worth storing separately.

        The upstream vessel decays toward the downstream all run, so its mean
        collapses onto the pore pressure -- close enough to it to be useless
        for setting the rig up, and indistinguishable from the outlet's mean.
        """
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        summary = loop.summarize()
        initial = summary.pulse_decay.initial_upstream_pressure_atm
        assert summary.mean_inlet_pressure_atm == pytest.approx(10.0, abs=0.2)
        assert initial > summary.mean_inlet_pressure_atm

    def test_the_pulse_is_the_difference_of_the_captured_pair(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """Both taken at one instant, so dP0 really is their difference."""
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        result = loop.summarize().pulse_decay
        span = (
            result.initial_upstream_pressure_atm
            - result.initial_downstream_pressure_atm
        )
        assert span == pytest.approx(result.pulse_amplitude_atm, rel=0.02)

    def test_the_storage_correction_is_applied_and_recorded(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        result = loop.summarize().pulse_decay
        assert result.storage_correction == "dicker_smits"
        assert result.storage_root is not None
        assert result.upstream_storage_ratio == pytest.approx(
            pulse_config.geometry().area_cm2 * 5.0 * 0.10 / 8.0, rel=1e-6
        )

    def test_brace_is_used_when_porosity_is_unrecorded(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        pulse_config.sample.porosity = None
        rate = 0.2
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        result = loop.summarize().pulse_decay
        assert result.storage_correction == "brace"
        assert result.storage_root is None

    def test_no_flow_is_recorded_at_all(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """None, not zero: no meter was read, and a zero would claim one was."""
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        assert all(r.flow_cm3_s is None for r in loop.readings)
        assert all(r.flow_voltage is None for r in loop.readings)
        assert loop.summarize().mean_flow_cm3_s is None

    def test_the_run_stops_at_the_configured_fraction(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=120.0,
        )
        assert "decay reached" in loop.stop_reason
        assert loop.status.decay_fraction <= pulse_config.run.pulse_decay.stop_below_fraction

    def test_a_run_with_no_pulse_says_so_and_is_not_confirmed(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=0.2, duration_s=10.0, pulse_atm=1e-6,
        )
        summary = loop.summarize()
        assert not summary.measurement_confirmed
        assert any("No pulse was ever detected" in w for w in summary.warnings)

    def test_the_console_line_and_header_render(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=20.0,
        )
        line = format_pulse_reading_line(loop.readings[-1], loop.status, pulse_config)
        assert "dP" in line
        assert "dP/dP0" in pulse_console_header(pulse_config)


class TestBudget:
    def test_the_measurand_names_the_method(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        budget = loop.summarize().uncertainty
        assert budget.measurand == "apparent gas permeability (pulse decay)"

    def test_the_decay_rate_is_the_type_a_term(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        budget = loop.summarize().uncertainty
        alpha = next(c for c in budget.components if c.symbol == "alpha")
        assert alpha.evaluation_type == "A"
        assert alpha.relative_sensitivity == pytest.approx(1.0)

    def test_no_flow_term_appears(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """The whole point of the method: the flowmeter is not an input."""
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        symbols = {c.symbol for c in loop.summarize().uncertainty.components}
        assert "Q" not in symbols
        assert {"alpha", "mu", "c_g", "P_mean", "V1", "V2", "L", "d"} <= symbols

    def test_the_pressure_enters_only_through_the_compressibility(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        budget = loop.summarize().uncertainty
        pressure = next(c for c in budget.components if c.symbol == "P_mean")
        assert pressure.relative_sensitivity == pytest.approx(-1.0, abs=0.03)
        assert any("measures a RATE" in note for note in budget.notes)

    def test_brace_volume_sensitivities_sum_to_one(self, pulse_config):
        """Exact in the zero-storage form; the numeric branch is covered above."""
        from gasperm.models import SampleGeometry
        from gasperm.uncertainty import PulseDecayPoint, build_pulse_decay_budget

        point = PulseDecayPoint(
            permeability_darcy=1e-5,
            decay_rate_per_s=0.01,
            mean_pressure_atm=10.0,
            viscosity_cp=0.0178,
            gas_compressibility_per_atm=0.1,
            temperature_c=22.0,
            upstream_volume_cm3=400.0,
            downstream_volume_cm3=75.0,
        )
        budget = build_pulse_decay_budget(
            point,
            SampleGeometry(sample_id="x", length_cm=5.0, diameter_cm=3.81),
            pulse_config.hardware,
            pulse_config.run,
            decay_rate_relative_uncertainty=0.01,
        )
        by_symbol = {c.symbol: c for c in budget.components}
        total = (
            by_symbol["V1"].relative_sensitivity + by_symbol["V2"].relative_sensitivity
        )
        assert total == pytest.approx(1.0)
        assert by_symbol["L"].relative_sensitivity == pytest.approx(1.0)
        assert by_symbol["d"].relative_sensitivity == pytest.approx(-2.0)

    def test_correlated_vessels_increase_the_uncertainty(self, pulse_config):
        """Unlike P1/P2: both volumes enter with the same sign."""
        from gasperm.models import SampleGeometry
        from gasperm.uncertainty import PulseDecayPoint, build_pulse_decay_budget

        point = PulseDecayPoint(
            permeability_darcy=1e-5,
            decay_rate_per_s=0.01,
            mean_pressure_atm=10.0,
            viscosity_cp=0.0178,
            gas_compressibility_per_atm=0.1,
            temperature_c=22.0,
            upstream_volume_cm3=100.0,
            downstream_volume_cm3=100.0,
        )
        geometry = SampleGeometry(sample_id="x", length_cm=5.0, diameter_cm=3.81)
        independent = build_pulse_decay_budget(
            point, geometry, pulse_config.hardware, pulse_config.run,
            decay_rate_relative_uncertainty=0.0,
        )
        pulse_config.hardware.reservoirs.correlation = 1.0
        correlated = build_pulse_decay_budget(
            point, geometry, pulse_config.hardware, pulse_config.run,
            decay_rate_relative_uncertainty=0.0,
        )
        assert correlated.correlation_relative_variance > 0.0
        assert (
            correlated.relative_combined_standard_uncertainty
            > independent.relative_combined_standard_uncertainty
        )


class TestStorageRoundTrip:
    def test_a_pulse_run_survives_the_sidecar(
        self, pulse_config, fake_analog_source, fake_temperature_source, tmp_path
    ):
        pulse_config.run.output_dir = str(tmp_path)
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        writer = RunWriter(pulse_config)
        writer.open()
        for reading in loop.readings:
            writer.write(reading)
        writer.close()
        summary = loop.summarize(csv_path=str(writer.readings_path))
        writer.write_metadata(summary)

        point = point_from_run(writer.directory)
        assert point.method == "pulse_decay"
        assert point.apparent_permeability_darcy == pytest.approx(
            summary.permeability_darcy, rel=1e-9
        )
        assert point.standard_uncertainty_darcy > 0.0

    def test_the_differential_is_stored_in_its_own_column(
        self, pulse_config, fake_analog_source, fake_temperature_source, tmp_path
    ):
        pulse_config.run.output_dir = str(tmp_path)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=0.2, duration_s=20.0,
        )
        writer = RunWriter(pulse_config)
        writer.open()
        for reading in loop.readings:
            writer.write(reading)
        writer.close()

        assert "delta_pressure_atm" in READING_COLUMNS
        assert "decay_fraction" in READING_COLUMNS
        rows = read_readings_csv(writer.readings_path)
        assert rows[-1]["delta_pressure_atm"] > 0.0
        assert rows[-1]["decay_fraction"] is not None
        # No meter was read, so the flow column is blank rather than zero.
        assert rows[-1]["flow_cm3_s"] is None

    def test_run_method_reads_the_block_not_the_key(self):
        """An old sidecar was steady-state by definition, not 'unknown'."""
        assert run_method({"run": {}}) == "steady_state"
        assert run_method({"run": {"method": "pulse_decay"}}) == "pulse_decay"
        assert run_method({}) is None
        assert run_method({"sample": {"id": "x"}}) is None

    def test_a_pulse_run_without_a_fit_is_refused_clearly(
        self, tmp_path, fake_run_writer
    ):
        """Not 'never reached steady state', which would be nonsense here."""
        directory = fake_run_writer(
            tmp_path, "core-001", datetime(2026, 3, 1, 9, tzinfo=timezone.utc),
            steady=False, method="pulse_decay",
        )
        with pytest.raises(ValueError, match="pulse-decay run with no recorded"):
            point_from_run(directory)

    def test_an_old_sidecar_still_short_circuits(self, tmp_path, fake_run_writer):
        """measurement_confirmed falls back to steady_state_reached."""
        directory = fake_run_writer(
            tmp_path, "core-001", datetime(2026, 3, 1, 9, tzinfo=timezone.utc)
        )
        assert point_from_run(directory).method == "steady_state"


class TestKlinkenbergMixing:
    def points(self, method: str):
        return [
            KlinkenbergPoint(
                mean_pressure_atm=p,
                apparent_permeability_darcy=5e-4 * (1.0 + 4.0 / p),
                sample_id="core-001",
                method=method,
                label=f"{method}-{p}",
            )
            for p in (5.0, 10.0, 20.0)
        ]

    def test_a_pulse_series_recovers_a_planted_k_l(self):
        result = fit_klinkenberg(self.points("pulse_decay"))
        assert result.liquid_permeability_darcy == pytest.approx(5e-4, rel=1e-9)
        assert result.slippage_factor_atm == pytest.approx(4.0, rel=1e-9)

    def test_mixing_methods_is_refused(self):
        mixed = self.points("pulse_decay")[:2] + self.points("steady_state")[2:]
        with pytest.raises(ValueError, match="same measurement method"):
            fit_klinkenberg(mixed)

    def test_the_refusal_explains_why(self):
        mixed = self.points("pulse_decay")[:2] + self.points("steady_state")[2:]
        with pytest.raises(ValueError, match="masquerade as slippage"):
            fit_klinkenberg(mixed)

    def test_mixing_can_be_allowed_and_then_warns(self):
        mixed = self.points("pulse_decay")[:2] + self.points("steady_state")[2:]
        result = fit_klinkenberg(mixed, allow_mixed_methods=True)
        assert any("different measurement methods" in w for w in result.warnings)

    def test_points_without_a_method_do_not_trigger_it(self):
        """CSV-loaded points carry no method; that is not a mixed set."""
        result = fit_klinkenberg(
            [
                KlinkenbergPoint(mean_pressure_atm=p, apparent_permeability_darcy=k)
                for p, k in ((5.0, 9e-4), (10.0, 7e-4), (20.0, 6e-4))
            ]
        )
        assert result.liquid_permeability_darcy > 0.0

    def test_describe_method_is_readable(self):
        assert describe_method("pulse_decay") == "pulse decay"
        assert describe_method(None) == "unknown"


class TestSummariseWithoutALoop:
    def test_summarize_is_usable_standalone(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """So a stored run can be re-reduced without re-running the rig."""
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        direct = summarize_pulse_decay_run(
            loop.readings, pulse_config, fit=loop.fit(), processor=loop.processor
        )
        assert direct.permeability_darcy == pytest.approx(
            loop.summarize().permeability_darcy
        )

    def test_an_empty_run_is_refused(self, pulse_config):
        with pytest.raises(ValueError, match="No samples"):
            summarize_pulse_decay_run([], pulse_config, fit=None)


class TestLivePlotPanels:
    def test_pulse_panels_replace_the_flow_panel(self, pulse_config):
        import matplotlib

        matplotlib.use("Agg")
        from gasperm.plotting import _panels_for

        keys = [panel.key for panel in _panels_for(pulse_config)]
        assert "flow" not in keys
        assert "delta_pressure" in keys
        assert "decay_fraction" in keys

    def test_steady_state_keeps_flow_and_drops_the_decay_panels(self, base_config):
        from gasperm.plotting import _panels_for

        keys = [panel.key for panel in _panels_for(base_config)]
        assert "flow" in keys
        assert "delta_pressure" not in keys

    def test_the_decay_fraction_panel_is_logarithmic(self, pulse_config):
        """An exponential decay reads as a straight line, which is the check."""
        from gasperm.plotting import _panels_for

        panel = next(p for p in _panels_for(pulse_config) if p.key == "decay_fraction")
        assert panel.yscale == "log"

    def test_the_decay_plot_is_written(
        self, pulse_config, fake_analog_source, fake_temperature_source, tmp_path
    ):
        import matplotlib

        matplotlib.use("Agg")
        from gasperm.plotting import plot_pulse_decay

        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        loop = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        )
        summary = loop.summarize()
        saved = plot_pulse_decay(
            summary.pulse_decay, loop.readings, path=tmp_path / "decay_fit.png"
        )
        assert saved.is_file()
        assert saved.stat().st_size > 0


class TestSpacers:
    """Hollow spacers upstream of the core: bore on the bench, length per run.

    A spacer is two measurements established in different places -- the bore is
    a property of a set of parts, the length of the one you just fitted -- so
    the tests below check that split holds all the way through.
    """

    def fittings(self, *specs):
        from gasperm.config import SpacerFitting

        return [SpacerFitting(type=name, length=length) for name, length in specs]

    def test_the_volume_splits_into_vessel_and_dead(self):
        reservoirs = GaspermConfig().hardware.reservoirs
        assert reservoirs.upstream.vessel == pytest.approx(380.0)
        assert reservoirs.upstream.dead == pytest.approx(20.0)
        assert reservoirs.upstream.fixed_volume_cm3 == pytest.approx(400.0)

    def test_a_reservoir_with_no_volume_at_all_is_refused(self):
        from gasperm.config import ReservoirConfig

        with pytest.raises(ValueError, match="positive total volume"):
            ReservoirConfig(vessel=0.0, dead=0.0)

    def test_a_spacer_volume_is_the_cylinder_of_its_bore_and_length(self):
        types = GaspermConfig().hardware.reservoirs.spacer_types
        wide = types["wide"]
        # 25.4 mm bore, 50 mm long: pi (1.27 cm)^2 * 5 cm.
        assert wide.volume_cm3(50.0) == pytest.approx(math.pi * 1.27**2 * 5.0, rel=1e-9)

    def test_the_two_bores_give_different_volumes_for_one_length(self):
        """The whole point of having types: bore enters squared."""
        types = GaspermConfig().hardware.reservoirs.spacer_types
        wide = types["wide"].volume_cm3(50.0)
        narrow = types["narrow"].volume_cm3(50.0)
        assert wide == pytest.approx(4.0 * narrow, rel=1e-9)

    def test_length_varies_within_a_type(self):
        wide = GaspermConfig().hardware.reservoirs.spacer_types["wide"]
        assert wide.volume_cm3(100.0) == pytest.approx(2.0 * wide.volume_cm3(50.0))

    def test_an_end_correction_is_added(self):
        """For chamfers and o-ring grooves the plain cylinder misses."""
        from gasperm.config import SpacerTypeConfig

        plain = SpacerTypeConfig(internal_diameter=25.4)
        corrected = SpacerTypeConfig(internal_diameter=25.4, end_correction_cm3=-0.3)
        assert corrected.volume_cm3(50.0) == pytest.approx(plain.volume_cm3(50.0) - 0.3)

    def test_a_mixed_stack_sums(self):
        reservoirs = GaspermConfig().hardware.reservoirs
        stack = self.fittings(("wide", 50.0), ("narrow", 30.0), ("wide", 25.0))
        expected = (
            reservoirs.spacer_types["wide"].volume_cm3(50.0)
            + reservoirs.spacer_types["narrow"].volume_cm3(30.0)
            + reservoirs.spacer_types["wide"].volume_cm3(25.0)
        )
        assert reservoirs.spacer_volume_cm3(stack) == pytest.approx(expected)
        assert reservoirs.upstream_volume_cm3(stack) == pytest.approx(400.0 + expected)

    def test_spacers_do_not_touch_v2(self):
        """They sit upstream of the core face."""
        reservoirs = GaspermConfig().hardware.reservoirs
        assert reservoirs.downstream_volume_cm3() == pytest.approx(75.0)

    def test_an_unknown_bore_is_refused_by_name(self):
        reservoirs = GaspermConfig().hardware.reservoirs
        with pytest.raises(ValueError, match="not defined"):
            reservoirs.resolve_spacer_type("enormous")
        with pytest.raises(ValueError, match="wide"):
            reservoirs.resolve_spacer_type("enormous")

    def test_bore_error_sums_within_a_type_and_lengths_add_in_quadrature(self):
        """The two measurements correlate differently, so they combine differently."""
        reservoirs = GaspermConfig().hardware.reservoirs
        wide = reservoirs.spacer_types["wide"]
        stack = self.fittings(("wide", 50.0), ("wide", 50.0))

        total_volume = 2 * wide.volume_cm3(50.0)
        from_bore = 2.0 * wide.relative_diameter_uncertainty() * total_volume
        from_lengths = wide.length_uncertainty_cm3() * math.sqrt(2)
        assert reservoirs.spacer_uncertainty_cm3(stack) == pytest.approx(
            math.hypot(from_bore, from_lengths), rel=1e-9
        )

    def test_treating_the_bore_as_independent_would_understate_it(self):
        """Why the split matters: it is not a per-spacer quadrature sum."""
        reservoirs = GaspermConfig().hardware.reservoirs
        wide = reservoirs.spacer_types["wide"]
        four = self.fittings(*[("wide", 50.0)] * 4)
        one_alone = reservoirs.spacer_uncertainty_cm3(self.fittings(("wide", 50.0)))
        naive = one_alone * math.sqrt(4)
        assert reservoirs.spacer_uncertainty_cm3(four) > naive
        assert wide.relative_diameter_uncertainty() > 0.0

    def test_different_bores_are_independent_of_each_other(self):
        reservoirs = GaspermConfig().hardware.reservoirs
        mixed = reservoirs.spacer_uncertainty_cm3(
            self.fittings(("wide", 50.0), ("narrow", 50.0))
        )
        wide_only = reservoirs.spacer_uncertainty_cm3(self.fittings(("wide", 50.0)))
        narrow_only = reservoirs.spacer_uncertainty_cm3(self.fittings(("narrow", 50.0)))
        assert mixed == pytest.approx(math.hypot(wide_only, narrow_only), rel=1e-9)

    def test_an_unknown_bore_in_the_run_is_fatal(self):
        """Fatal, not a warning: the volume cannot be guessed."""
        from gasperm.config import ConfigError, validate_for_collect

        config = GaspermConfig()
        config.run.method = "pulse_decay"
        config.hardware.temperature.required = False
        config.run.pulse_decay.upstream_spacers = self.fittings(("enormous", 50.0))
        with pytest.raises(ConfigError, match="not defined"):
            validate_for_collect(config)

    def test_the_stack_is_reported_at_startup(self):
        from gasperm.config import validate_for_collect

        config = GaspermConfig()
        config.run.method = "pulse_decay"
        config.hardware.temperature.required = False
        config.run.pulse_decay.upstream_spacers = self.fittings(
            ("wide", 50.0), ("narrow", 30.0)
        )
        notes = [w for w in validate_for_collect(config) if "spacer" in w]
        assert any("wide:50" in w and "narrow:30" in w for w in notes)

    def test_the_breakdown_is_printable(self):
        reservoirs = GaspermConfig().hardware.reservoirs
        text = reservoirs.describe(self.fittings(("wide", 50.0), ("narrow", 30.0)))
        assert "vessel 380" in text
        assert "dead 20" in text
        assert "wide 50mm" in text
        assert "narrow 30mm" in text

    def test_spacers_change_the_recovered_permeability(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """The physical point: V1 is in the equation, so the stack matters.

        The same decay analysed with a different stack gives a different k --
        which is why an unrecorded spacer is a silent error, and why the stack
        goes on the record.
        """
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)

        pulse_config.run.pulse_decay.upstream_spacers = []
        bare = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        ).summarize()

        pulse_config.run.pulse_decay.upstream_spacers = self.fittings(
            ("narrow", 40.0), ("narrow", 40.0)
        )
        stacked = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        ).summarize()

        assert stacked.pulse_decay.upstream_spacers == ["narrow:40", "narrow:40"]
        assert stacked.pulse_decay.spacer_volume_cm3 > 0.0
        assert stacked.permeability_darcy > bare.permeability_darcy

    def test_the_stack_reaches_the_budget(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        stack = self.fittings(("narrow", 30.0), ("narrow", 30.0), ("narrow", 30.0))
        pulse_config.run.pulse_decay.upstream_spacers = stack
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        summary = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        ).summarize()
        budget = summary.uncertainty
        upstream = next(c for c in budget.components if c.symbol == "V1")
        expected = pulse_config.hardware.reservoirs.upstream_uncertainty_cm3(stack)
        assert upstream.standard_uncertainty == pytest.approx(expected, rel=1e-9)
        assert any("narrow:30" in note for note in budget.notes)

    def test_no_spacers_leaves_the_budget_unchanged(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        summary = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        ).summarize()
        budget = summary.uncertainty
        upstream = next(c for c in budget.components if c.symbol == "V1")
        assert upstream.standard_uncertainty == pytest.approx(
            pulse_config.hardware.reservoirs.upstream.fixed_uncertainty_cm3, rel=1e-9
        )
        # The general "plus any upstream spacers" note always appears; what
        # must not is a note describing an actual stack.
        assert not any("add " in note and "cm3 to V1" in note for note in budget.notes)


class TestLeakTest:
    """The pre-step: characterise the rig before believing anything about rock.

    A leak produces a differential decay indistinguishable from a slow sample,
    so a pulse-decay measurement means nothing without a bound on it. These
    check that the bound is recorded, found again, compared, and never mistaken
    for a measurement.
    """

    def leak_config(self, pulse_config):
        pulse_config.run.purpose = "leak_test"
        pulse_config.run.pulse_decay.leak_test_duration_s = 30.0
        return pulse_config

    def write_run(self, config, loop, tmp_path):
        """Store a completed run the way `collect` does, and return its dir."""
        from gasperm.storage import RunWriter

        config.run.output_dir = str(tmp_path)
        writer = RunWriter(config)
        writer.open()
        for reading in loop.readings:
            writer.write(reading)
        writer.close()
        writer.write_metadata(loop.summarize(csv_path=str(writer.readings_path)))
        return writer.directory

    # -- what a leak test is ------------------------------------------------

    def test_a_leak_test_only_makes_sense_for_pulse_decay(self):
        """Steady-state on a blanked plug would just report no flow."""
        from gasperm.config import RunConfig

        with pytest.raises(ValueError, match="pulse-decay observation"):
            RunConfig(purpose="leak_test", method="steady_state")

    def test_it_is_marked_in_the_summary(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        config = self.leak_config(pulse_config)
        summary = run_decay(
            config, fake_analog_source, fake_temperature_source,
            rate=0.05, duration_s=30.0,
        ).summarize()
        assert summary.purpose == "leak_test"
        assert summary.method == "pulse_decay"

    def test_it_runs_for_its_duration_rather_than_to_a_decay_fraction(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """On a tight rig nothing decays, so there is no completion signal."""
        config = self.leak_config(pulse_config)
        loop = run_decay(
            config, fake_analog_source, fake_temperature_source,
            rate=1e-6, duration_s=20.0,
        )
        assert "decay reached" not in loop.stop_reason

    def test_no_decay_is_a_pass_not_a_failure(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """The inversion that matters: for a measurement this is a failure."""
        config = self.leak_config(pulse_config)
        summary = run_decay(
            config, fake_analog_source, fake_temperature_source,
            rate=1e-9, duration_s=20.0,
        ).summarize()
        assert summary.pulse_decay is None
        assert summary.measurement_confirmed
        assert any("result you want" in w for w in summary.warnings)

    def test_the_same_state_is_a_failure_for_a_measurement(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        summary = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=1e-9, duration_s=20.0,
        ).summarize()
        assert not summary.measurement_confirmed
        assert any("did NOT measure" in w for w in summary.warnings)

    def test_a_measurable_leak_is_reported_as_a_permeability(
        self, pulse_config, fake_analog_source, fake_temperature_source
    ):
        """The number to compare against k, in the units of the decision."""
        config = self.leak_config(pulse_config)
        summary = run_decay(
            config, fake_analog_source, fake_temperature_source,
            rate=0.05, duration_s=30.0,
        ).summarize()
        assert summary.permeability_darcy > 0.0
        assert any("LEAK TEST:" in w for w in summary.warnings)
        assert any("floor for a trustworthy measurement" in w for w in summary.warnings)

    # -- how a measurement uses it -----------------------------------------

    def test_a_measurement_without_one_says_so(
        self, pulse_config, fake_analog_source, fake_temperature_source, tmp_path
    ):
        pulse_config.run.output_dir = str(tmp_path)
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        summary = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        ).summarize()
        assert summary.pulse_decay.leak_rate_per_s is None
        assert any("No leak test has been recorded" in w for w in summary.warnings)

    def test_a_recorded_leak_is_found_and_compared(
        self, pulse_config, fake_analog_source, fake_temperature_source, tmp_path
    ):
        sample_rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)

        leak = self.leak_config(pulse_config)
        leak_loop = run_decay(
            leak, fake_analog_source, fake_temperature_source,
            rate=sample_rate * 0.3, duration_s=30.0,
        )
        self.write_run(leak, leak_loop, tmp_path)

        pulse_config.run.purpose = "measurement"
        summary = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=sample_rate, duration_s=40.0,
        ).summarize()

        result = summary.pulse_decay
        assert result.leak_rate_per_s == pytest.approx(sample_rate * 0.3, rel=0.05)
        assert result.leak_equivalent_permeability_darcy > 0.0
        assert result.leak_fraction == pytest.approx(0.3, abs=0.02)
        assert any("rig's own decay is" in w for w in summary.warnings)

    def test_a_small_leak_does_not_warn(
        self, pulse_config, fake_analog_source, fake_temperature_source, tmp_path
    ):
        sample_rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        leak = self.leak_config(pulse_config)
        self.write_run(
            leak,
            run_decay(
                leak, fake_analog_source, fake_temperature_source,
                rate=sample_rate * 0.01, duration_s=30.0,
            ),
            tmp_path,
        )
        pulse_config.run.purpose = "measurement"
        summary = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=sample_rate, duration_s=40.0,
        ).summarize()
        assert not any("above pulse_decay.max_leak_fraction" in w for w in summary.warnings)

    def test_a_passing_leak_test_is_not_the_same_as_none(
        self, pulse_config, fake_analog_source, fake_temperature_source, tmp_path
    ):
        """Telling an operator who did the right thing that they did not is worse
        than saying nothing."""
        leak = self.leak_config(pulse_config)
        self.write_run(
            leak,
            run_decay(
                leak, fake_analog_source, fake_temperature_source,
                rate=1e-9, duration_s=20.0,
            ),
            tmp_path,
        )
        pulse_config.run.purpose = "measurement"
        rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        summary = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=rate, duration_s=40.0,
        ).summarize()
        assert not any("No leak test has been recorded" in w for w in summary.warnings)
        assert any("no measurable decay" in w for w in summary.warnings)

    def test_the_leak_can_be_subtracted_when_asked(
        self, pulse_config, fake_analog_source, fake_temperature_source, tmp_path
    ):
        sample_rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        leak = self.leak_config(pulse_config)
        self.write_run(
            leak,
            run_decay(
                leak, fake_analog_source, fake_temperature_source,
                rate=sample_rate * 0.3, duration_s=30.0,
            ),
            tmp_path,
        )
        pulse_config.run.purpose = "measurement"

        uncorrected = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=sample_rate, duration_s=40.0,
        ).summarize()

        pulse_config.run.pulse_decay.leak_correction = "subtract"
        corrected = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=sample_rate, duration_s=40.0,
        ).summarize()

        assert corrected.pulse_decay.leak_subtracted
        assert corrected.permeability_darcy < uncorrected.permeability_darcy
        assert any("was SUBTRACTED" in w for w in corrected.warnings)

    def test_subtracting_is_off_by_default(self, pulse_config):
        """A leak that changed since the test would corrupt a result silently."""
        assert pulse_config.run.pulse_decay.leak_correction == "off"

    def test_a_leak_test_at_a_different_pressure_is_flagged(
        self, pulse_config, fake_analog_source, fake_temperature_source, tmp_path
    ):
        """Leak conductance is pressure-dependent, so the test must match."""
        sample_rate = true_decay_rate(pulse_config, 5.0e-4, 10.0)
        leak = self.leak_config(pulse_config)
        self.write_run(
            leak,
            run_decay(
                leak, fake_analog_source, fake_temperature_source,
                rate=sample_rate * 0.01, duration_s=30.0, mean_pressure_atm=3.0,
            ),
            tmp_path,
        )
        pulse_config.run.purpose = "measurement"
        summary = run_decay(
            pulse_config, fake_analog_source, fake_temperature_source,
            rate=sample_rate, duration_s=40.0, mean_pressure_atm=10.0,
        ).summarize()
        assert any("does not describe this run" in w for w in summary.warnings)

    # -- it is never a data point ------------------------------------------

    def test_a_leak_test_is_refused_as_a_klinkenberg_point(
        self, tmp_path, fake_run_writer
    ):
        from gasperm.storage import point_from_run

        directory = fake_run_writer(
            tmp_path, "core-001", datetime(2026, 3, 1, 9, tzinfo=timezone.utc),
            method="pulse_decay", purpose="leak_test",
        )
        with pytest.raises(ValueError, match="leak test, not a measurement"):
            point_from_run(directory)

    def test_run_purpose_reads_the_block_not_the_key(self):
        from gasperm.storage import run_purpose

        assert run_purpose({"run": {}}) == "measurement"
        assert run_purpose({"run": {"purpose": "leak_test"}}) == "leak_test"
        assert run_purpose({}) is None

    def test_the_latest_leak_test_wins(self, tmp_path, fake_run_writer):
        from gasperm.storage import find_leak_test, find_runs

        fake_run_writer(
            tmp_path, "core-001", datetime(2026, 3, 1, 9, tzinfo=timezone.utc),
            method="pulse_decay", purpose="leak_test",
        )
        newest = fake_run_writer(
            tmp_path, "core-002", datetime(2026, 3, 2, 9, tzinfo=timezone.utc),
            method="pulse_decay", purpose="leak_test",
        )
        fake_run_writer(
            tmp_path, "core-001", datetime(2026, 3, 3, 9, tzinfo=timezone.utc),
            method="pulse_decay",
        )
        found = find_leak_test(find_runs(tmp_path))
        assert found.directory == newest

    def test_it_is_found_across_plugs(self, tmp_path, fake_run_writer):
        """The rig leaked the same whichever core was in it."""
        from gasperm.storage import find_leak_test, find_runs

        fake_run_writer(
            tmp_path, "core-999", datetime(2026, 3, 1, 9, tzinfo=timezone.utc),
            method="pulse_decay", purpose="leak_test",
        )
        assert find_leak_test(find_runs(tmp_path)) is not None


class TestConfigRefusals:
    def test_a_supplied_p2_and_pulse_decay_cannot_coexist(self):
        from gasperm.config import RunConfig

        with pytest.raises(ValueError, match="CLOSED downstream vessel"):
            RunConfig(method="pulse_decay", downstream_pressure=101.325)

    def test_a_meterless_rig_loads_for_pulse_decay(self):
        from gasperm.config import HardwareConfig, RunConfig

        config = GaspermConfig(
            hardware=HardwareConfig(flowmeters={}, default_flowmeter=None),
            run=RunConfig(method="pulse_decay"),
        )
        # experiment_metadata is on the pulse path twice and must not need one.
        from gasperm.config import experiment_metadata

        assert experiment_metadata(config).flowmeter == ""

    def test_the_effective_volume_is_the_harmonic_combination(self):
        """It is the SMALLER vessel that sets the decay rate."""
        config = GaspermConfig()
        reservoirs = config.hardware.reservoirs
        assert reservoirs.effective_volume_cm3() == pytest.approx(
            400.0 * 75.0 / 475.0, rel=1e-9
        )
        assert reservoirs.effective_volume_cm3() < 75.0

    def test_the_predicted_duration_is_reported_at_startup(self):
        from gasperm.config import validate_for_collect

        config = GaspermConfig()
        config.run.method = "pulse_decay"
        config.hardware.temperature.required = False
        config.sample.porosity = 0.10
        config.run.pulse_decay.expected_permeability = 1.0
        warnings = validate_for_collect(config)
        timing = [w for w in warnings if "time constant" in w]
        assert len(timing) == 1
        # 13.9 h at 10 atm on this rig's 400/75 cm3 vessels.
        assert "13.9 h" in timing[0]

    def test_a_pulse_below_the_transducer_resolution_is_flagged(self):
        from gasperm.config import validate_for_collect

        config = GaspermConfig()
        config.run.method = "pulse_decay"
        config.hardware.temperature.required = False
        warnings = validate_for_collect(config)
        assert any("standard uncertainty of" in w for w in warnings)


def test_the_module_imports_no_hardware():
    """The physics must stay testable with no device attached."""
    import gasperm.pulse_decay as module

    source = (
        module.__file__.replace("pulse_decay.py", "pulse_decay.py")
    )
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "import nidaqmx" not in text
    assert "import serial" not in text
    assert math.isfinite(1.0)
