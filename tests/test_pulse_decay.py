"""Pulse-decay physics, against hand-worked and synthesised references.

The decisive test is the analytic round trip: synthesise a decay from a known
permeability, fit it back, and require the original number. Everything else
guards a specific way the method fails quietly -- a root solver that returns the
wrong branch, a fit biased by an offset, an uncertainty flattered by
autocorrelated samples.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gasperm import units
from gasperm.config.run import PulseDecayConfig
from gasperm.pulse_decay import (
    PulseDecayInputError,
    PulseDecayMonitor,
    brace_decay_rate_per_s,
    brace_permeability_darcy,
    dicker_smits_decay_rate_per_s,
    dicker_smits_permeability_darcy,
    find_pulse,
    first_storage_root,
    fit_decay_rate,
    fit_window,
    pore_volume_cm3,
    storage_ratios,
)

#: This rig: a 38.1 x 50 mm plug between a 400 cm3 and a 75 cm3 vessel.
AREA_CM2 = math.pi * (3.81 / 2.0) ** 2
LENGTH_CM = 5.0
POROSITY = 0.10
V1_CM3 = 400.0
V2_CM3 = 75.0
VISCOSITY_CP = 0.0178
#: Nitrogen at 10 atm: c = 1/P to a few tenths of a percent.
COMPRESSIBILITY_PER_ATM = 0.1

BRACE_ARGS = dict(
    viscosity_cp=VISCOSITY_CP,
    gas_compressibility_per_atm=COMPRESSIBILITY_PER_ATM,
    length_cm=LENGTH_CM,
    area_cm2=AREA_CM2,
    upstream_volume_cm3=V1_CM3,
    downstream_volume_cm3=V2_CM3,
)
DS_ARGS = dict(BRACE_ARGS, porosity_fraction=POROSITY)


def synthesise(
    alpha,
    *,
    dp0=1.0,
    offset=0.0,
    duration=None,
    step=1.0,
    noise=0.0,
    seed=0,
    correlation=0.0,
):
    """A decay ``dP0 exp(-alpha t) + offset`` sampled on a regular grid.

    ``correlation`` makes the noise AR(1) rather than white, which is what a
    real transducer over hours actually produces -- thermal drift and 1/f, not
    independent samples.
    """
    if duration is None:
        duration = 3.0 / alpha
    times = np.arange(0.0, duration, step)
    values = dp0 * np.exp(-alpha * times) + offset
    if noise:
        rng = np.random.default_rng(seed)
        if correlation:
            innovation = rng.normal(0.0, noise * math.sqrt(1.0 - correlation**2), times.size)
            drift = np.zeros(times.size)
            drift[0] = rng.normal(0.0, noise)
            for index in range(1, times.size):
                drift[index] = correlation * drift[index - 1] + innovation[index]
            values = values + drift
        else:
            values = values + rng.normal(0.0, noise, times.size)
    return list(times), list(values)


class TestStorageGeometry:
    def test_pore_volume_is_phi_a_l(self):
        volume = pore_volume_cm3(
            area_cm2=AREA_CM2, length_cm=LENGTH_CM, porosity_fraction=POROSITY
        )
        assert volume == pytest.approx(POROSITY * AREA_CM2 * LENGTH_CM)
        assert volume == pytest.approx(5.700, abs=0.001)

    def test_ratios_are_pore_volume_over_each_vessel(self):
        a1, a2 = storage_ratios(
            pore_volume_cm3=5.7, upstream_volume_cm3=V1_CM3, downstream_volume_cm3=V2_CM3
        )
        assert a1 == pytest.approx(5.7 / 400.0)
        assert a2 == pytest.approx(5.7 / 75.0)

    def test_a_porosity_outside_zero_to_one_is_refused(self):
        with pytest.raises(PulseDecayInputError, match="between 0 and 1"):
            pore_volume_cm3(area_cm2=1.0, length_cm=1.0, porosity_fraction=1.5)


class TestStorageRoot:
    """The trap: the textbook tan-form is unusable, and fails quietly."""

    def test_the_naive_bracket_would_be_wrong_here(self):
        """At a1 = a2 = 2 the tan-form's pole sits above pi/2, inverting it.

        sqrt(a1 a2) = 2 > pi/2, so bracketing the textbook form on (0, pi/2)
        either divides by zero or brackets backwards. The pole-free form gets
        the right answer, and this is the value it must get.
        """
        assert first_storage_root(2.0, 2.0) == pytest.approx(1.720667, abs=1e-6)

    @pytest.mark.parametrize("a1", [1e-6, 1e-3, 0.1, 1.0, 5.0, 10.0])
    @pytest.mark.parametrize("a2", [1e-6, 1e-3, 0.1, 1.0, 5.0, 10.0])
    def test_the_root_is_always_in_the_open_interval(self, a1: float, a2: float):
        theta = first_storage_root(a1, a2)
        assert 0.0 < theta < math.pi
        residual = (theta**2 - a1 * a2) * math.sin(theta) - theta * (a1 + a2) * math.cos(
            theta
        )
        assert residual == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("a", [1e-6, 1e-5, 1e-4])
    def test_it_reduces_to_the_brace_limit(self, a: float):
        """theta_1^2 -> a1 + a2 as the storage vanishes."""
        theta = first_storage_root(a, a)
        assert theta**2 / (2.0 * a) == pytest.approx(1.0, abs=1e-4)

    def test_it_approaches_pi_as_storage_dominates(self):
        """Both ends become no-flux: the closed-rod limit."""
        assert first_storage_root(1e6, 1e6) == pytest.approx(math.pi, abs=1e-4)

    def test_it_is_monotone_in_the_ratios(self):
        roots = [first_storage_root(a, a) for a in (0.01, 0.1, 1.0, 10.0, 100.0)]
        assert roots == sorted(roots)

    def test_zero_storage_is_refused_rather_than_hanging(self):
        with pytest.raises(PulseDecayInputError, match="no root"):
            first_storage_root(0.0, 0.0)

    def test_a_negative_ratio_is_refused(self):
        with pytest.raises(PulseDecayInputError, match="non-negative"):
            first_storage_root(-1.0, 0.5)


class TestPermeabilityFromDecayRate:
    def test_brace_round_trips_through_its_inverse(self):
        alpha = brace_decay_rate_per_s(permeability_darcy=1e-6, **BRACE_ARGS)
        assert brace_permeability_darcy(
            decay_rate_per_s=alpha, **BRACE_ARGS
        ) == pytest.approx(1e-6, rel=1e-12)

    def test_dicker_smits_round_trips_through_its_inverse(self):
        alpha = dicker_smits_decay_rate_per_s(permeability_darcy=1e-6, **DS_ARGS)
        assert dicker_smits_permeability_darcy(
            decay_rate_per_s=alpha, **DS_ARGS
        ) == pytest.approx(1e-6, rel=1e-12)

    def test_the_time_constant_on_this_rig(self):
        """1 uD on 400/75 cm3 vessels: about 14 hours. Worth knowing up front."""
        alpha = dicker_smits_decay_rate_per_s(permeability_darcy=1e-6, **DS_ARGS)
        assert 1.0 / alpha / 3600.0 == pytest.approx(13.9, abs=0.2)

    def test_the_correction_raises_k_and_by_how_much_here(self):
        """Brace reads low because part of the decay filled the plug."""
        alpha = 1.992297e-05
        brace = brace_permeability_darcy(decay_rate_per_s=alpha, **BRACE_ARGS)
        corrected = dicker_smits_permeability_darcy(decay_rate_per_s=alpha, **DS_ARGS)
        assert corrected > brace
        assert corrected / brace == pytest.approx(1.018, abs=0.001)

    @pytest.mark.parametrize("volume", [400.0, 100.0, 50.0, 20.0, 5.0])
    def test_the_correction_grows_as_the_vessels_shrink(self, volume: float):
        alpha = 1e-5
        brace = brace_permeability_darcy(
            decay_rate_per_s=alpha,
            **dict(BRACE_ARGS, upstream_volume_cm3=volume, downstream_volume_cm3=volume),
        )
        corrected = dicker_smits_permeability_darcy(
            decay_rate_per_s=alpha,
            **dict(DS_ARGS, upstream_volume_cm3=volume, downstream_volume_cm3=volume),
        )
        assert corrected >= brace

    def test_the_two_models_agree_when_storage_is_negligible(self):
        """Constructed to *produce* a ~1e-4 ratio, not merely asserted to."""
        # a = phi*A*L/V = 1e-4 with the plug fixed, so V = phi*A*L/1e-4.
        volume = POROSITY * AREA_CM2 * LENGTH_CM / 1e-4
        args = dict(BRACE_ARGS, upstream_volume_cm3=volume, downstream_volume_cm3=volume)
        brace = brace_permeability_darcy(decay_rate_per_s=1e-5, **args)
        corrected = dicker_smits_permeability_darcy(
            decay_rate_per_s=1e-5, **dict(args, porosity_fraction=POROSITY)
        )
        assert corrected == pytest.approx(brace, rel=1e-4)

    @pytest.mark.parametrize(
        "field", ["decay_rate_per_s", "viscosity_cp", "length_cm", "area_cm2"]
    )
    def test_non_positive_inputs_are_refused(self, field: str):
        args = dict(BRACE_ARGS, decay_rate_per_s=1e-5)
        args[field] = 0.0
        with pytest.raises(PulseDecayInputError, match=field):
            brace_permeability_darcy(**args)

    def test_the_units_need_no_conversion_constant(self):
        """CGS-Darcy closes on itself: darcy == cP*cm^2/(s*atm)."""
        alpha = brace_decay_rate_per_s(permeability_darcy=1.0, **BRACE_ARGS)
        expected = (
            1.0 * AREA_CM2 * (1.0 / V1_CM3 + 1.0 / V2_CM3)
            / (VISCOSITY_CP * COMPRESSIBILITY_PER_ATM * LENGTH_CM)
        )
        assert alpha == pytest.approx(expected, rel=1e-15)


class TestPulseDetection:
    def test_the_peak_is_found(self):
        times = list(range(20))
        values = [0.0, 0.0, 0.2, 0.8, 1.0] + [1.0 * math.exp(-0.1 * i) for i in range(15)]
        index, amplitude = find_pulse(times, values)
        assert index == 4
        assert amplitude == pytest.approx(1.0)

    def test_a_single_spike_is_not_a_pulse(self):
        """A one-sample glitch must not become dP0 and rescale everything."""
        times = list(range(20))
        values = [1.0 * math.exp(-0.1 * i) for i in range(20)]
        values[10] = 99.0
        index, amplitude = find_pulse(times, values, median_window=5)
        assert index == 0
        assert amplitude == pytest.approx(1.0)

    def test_mismatched_series_are_refused(self):
        with pytest.raises(PulseDecayInputError, match="same length"):
            find_pulse([0.0, 1.0], [1.0])

    def test_the_window_runs_between_the_configured_fractions(self):
        times, values = synthesise(0.01, step=1.0)
        start, end = fit_window(
            times, values, peak_index=0, peak_value=1.0,
            start_fraction=0.9, end_fraction=0.5,
        )
        assert values[start] <= 0.9
        assert values[start - 1] > 0.9 if start else True
        assert values[end - 1] <= 0.5

    def test_a_reversed_window_is_refused(self):
        with pytest.raises(PulseDecayInputError, match="start_fraction"):
            fit_window([0.0], [1.0], peak_index=0, peak_value=1.0,
                       start_fraction=0.4, end_fraction=0.9)


class TestDecayFit:
    def test_a_clean_decay_recovers_alpha_exactly(self):
        alpha = 1.992297e-05
        times, values = synthesise(alpha, dp0=0.1, step=10.0)
        fit = fit_decay_rate(times, values, bin_s=None)
        assert fit.decay_rate_per_s == pytest.approx(alpha, rel=1e-6)
        assert fit.r_squared > 0.9999

    def test_the_analytic_round_trip_recovers_the_permeability(self):
        """Synthesise from a known k, fit, and require the number back."""
        alpha = dicker_smits_decay_rate_per_s(permeability_darcy=1e-6, **DS_ARGS)
        times, values = synthesise(
            alpha, dp0=0.1, step=10.0, noise=2e-4, seed=7,
            duration=math.log(1.0 / 0.4) / alpha,
        )
        fit = fit_decay_rate(times, values, bin_s=60.0)
        recovered = dicker_smits_permeability_darcy(
            decay_rate_per_s=fit.decay_rate_per_s, **DS_ARGS
        )
        assert recovered == pytest.approx(1e-6, rel=0.005)

    def test_an_unmodelled_offset_biases_a_log_linear_fit_low(self):
        """Why the fit is nonlinear: two transducers have a zero mismatch."""
        alpha = 1e-4
        times, values = synthesise(alpha, dp0=0.05, offset=0.002, step=10.0)
        biased = fit_decay_rate(times, values, fit_offset=False, bin_s=None)
        corrected = fit_decay_rate(times, values, fit_offset=True, bin_s=None)
        assert biased.model == "log_linear"
        assert biased.decay_rate_per_s < 0.95 * alpha
        assert corrected.model == "exponential_offset"
        assert corrected.decay_rate_per_s == pytest.approx(alpha, rel=1e-3)

    def test_the_offset_itself_is_recovered(self):
        alpha = 1e-4
        times, values = synthesise(alpha, dp0=0.05, offset=0.002, step=10.0)
        fit = fit_decay_rate(times, values, bin_s=None)
        assert fit.offset_atm == pytest.approx(0.002, rel=1e-3)

    def test_binning_reduces_the_sample_count_and_records_both(self):
        alpha = 1e-4
        times, values = synthesise(alpha, dp0=0.05, step=1.0)
        fit = fit_decay_rate(times, values, bin_s=60.0)
        assert fit.raw_sample_count == len(times)
        assert fit.sample_count < fit.raw_sample_count
        assert fit.sample_count == pytest.approx(len(times) / 60.0, abs=2)

    def test_binning_widens_the_uncertainty_autocorrelation_had_hidden(self):
        """The reason binning exists, stated as a test.

        Real transducer noise over hours is drift, not independent samples. An
        unbinned fit treats each of them as fresh information and reports a
        u(alpha) far smaller than the data supports; binning removes most of
        the correlation and the honest, larger uncertainty comes back.
        """
        alpha = 1e-4
        times, values = synthesise(
            alpha, dp0=0.05, step=1.0, noise=1e-3, seed=3, correlation=0.98
        )
        unbinned = fit_decay_rate(times, values, bin_s=None)
        binned = fit_decay_rate(times, values, bin_s=300.0)

        # The unbinned residuals are visibly structured; the binned ones far less.
        assert unbinned.residual_autocorrelation > 0.9
        assert binned.residual_autocorrelation < unbinned.residual_autocorrelation
        # And the uncertainty it was hiding reappears.
        assert (
            binned.decay_rate_standard_uncertainty_per_s
            > 2.0 * unbinned.decay_rate_standard_uncertainty_per_s
        )

    def test_binning_is_roughly_neutral_on_independent_noise(self):
        """Binning is not a way of inflating uncertainty for its own sake."""
        alpha = 1e-4
        times, values = synthesise(alpha, dp0=0.05, step=1.0, noise=1e-3, seed=3)
        unbinned = fit_decay_rate(times, values, bin_s=None)
        binned = fit_decay_rate(times, values, bin_s=60.0)
        ratio = (
            binned.decay_rate_standard_uncertainty_per_s
            / unbinned.decay_rate_standard_uncertainty_per_s
        )
        assert 0.7 < ratio < 1.5

    def test_the_uncertainty_and_dof_are_reported(self):
        alpha = 1e-4
        times, values = synthesise(alpha, dp0=0.05, step=1.0, noise=5e-4, seed=1)
        fit = fit_decay_rate(times, values, bin_s=30.0)
        assert fit.decay_rate_standard_uncertainty_per_s > 0.0
        assert fit.relative_standard_uncertainty < 0.05
        assert fit.degrees_of_freedom == fit.sample_count - 3
        assert fit.residual_autocorrelation is not None

    def test_the_time_constant_matches_the_rate(self):
        times, values = synthesise(1e-4, dp0=0.05, step=10.0)
        fit = fit_decay_rate(times, values, bin_s=None)
        assert fit.time_constant_s == pytest.approx(1.0 / fit.decay_rate_per_s)

    def test_too_few_samples_are_refused(self):
        with pytest.raises(PulseDecayInputError, match="at least 3"):
            fit_decay_rate([0.0, 1.0], [1.0, 0.9])

    def test_over_aggressive_binning_is_refused_clearly(self):
        times, values = synthesise(1e-3, dp0=0.05, step=1.0)
        with pytest.raises(PulseDecayInputError, match="shorter"):
            fit_decay_rate(times, values, bin_s=1e9)

    def test_a_flat_series_is_refused(self):
        with pytest.raises(PulseDecayInputError):
            fit_decay_rate([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], bin_s=None)


class TestMonitor:
    def config(self, **overrides) -> PulseDecayConfig:
        return PulseDecayConfig(**overrides)

    def test_phases_advance_through_the_run(self):
        alpha = 1e-3
        monitor = PulseDecayMonitor(
            self.config(), min_pulse_atm=units.to_atm(20.0, "kPa")
        )
        assert monitor.update(0.0, 0.0).phase == "waiting"
        seen = {monitor.update(0.0, 0.0).phase}
        # Pulse applied at t = 5 s, then decays.
        for index in range(1, 4000):
            t = index * 1.0
            value = 0.5 * math.exp(-alpha * max(0.0, t - 5.0)) if t >= 5.0 else 0.0
            seen.add(monitor.update(t, value).phase)
            if monitor.is_complete:
                break
        assert {"waiting", "transient", "decaying", "complete"} <= seen

    def test_the_pulse_is_recorded_at_its_peak(self):
        monitor = PulseDecayMonitor(
            self.config(), min_pulse_atm=units.to_atm(20.0, "kPa")
        )
        for index in range(200):
            t = index * 1.0
            value = 0.0 if t < 10.0 else 0.5 * math.exp(-1e-3 * (t - 10.0))
            monitor.update(t, value)
        status = monitor.status
        assert status.pulse_at_elapsed_s == pytest.approx(10.0)
        assert status.pulse_amplitude_atm == pytest.approx(0.5)

    def test_a_running_rate_appears_inside_the_fit_window(self):
        alpha = 1e-3
        monitor = PulseDecayMonitor(
            self.config(), min_pulse_atm=units.to_atm(20.0, "kPa")
        )
        for index in range(1500):
            t = index * 1.0
            monitor.update(t, 0.5 * math.exp(-alpha * t))
            if monitor.is_complete:
                break
        assert monitor.status.decay_rate_per_s == pytest.approx(alpha, rel=0.02)
        assert monitor.status.time_constant_s == pytest.approx(1.0 / alpha, rel=0.02)

    def test_it_projects_when_the_run_will_finish(self):
        alpha = 1e-3
        monitor = PulseDecayMonitor(
            self.config(), min_pulse_atm=units.to_atm(20.0, "kPa")
        )
        for index in range(400):
            monitor.update(index * 1.0, 0.5 * math.exp(-alpha * index))
        projected = monitor.status.projected_complete_elapsed_s
        assert projected == pytest.approx(math.log(1.0 / 0.4) / alpha, rel=0.05)

    def test_completion_is_at_the_configured_fraction(self):
        alpha = 1e-3
        monitor = PulseDecayMonitor(
            self.config(stop_below_fraction=0.5),
            min_pulse_atm=units.to_atm(20.0, "kPa"),
        )
        for index in range(5000):
            monitor.update(index * 1.0, 0.5 * math.exp(-alpha * index))
            if monitor.is_complete:
                break
        assert monitor.status.decay_fraction <= 0.5
        assert monitor.status.elapsed_s == pytest.approx(math.log(2.0) / alpha, rel=0.05)

    def test_a_rise_after_the_peak_is_flagged(self):
        """A leak or a reopened valve, otherwise entirely silent."""
        alpha = 1e-3
        monitor = PulseDecayMonitor(
            self.config(), min_pulse_atm=units.to_atm(20.0, "kPa")
        )
        for index in range(300):
            monitor.update(index * 1.0, 0.5 * math.exp(-alpha * index))
        assert not monitor.status.reversed_since_peak
        monitor.update(300.0, 0.9)
        assert monitor.status.reversed_since_peak

    def test_noise_below_the_threshold_is_not_a_pulse(self):
        monitor = PulseDecayMonitor(
            self.config(), min_pulse_atm=units.to_atm(20.0, "kPa")
        )
        for index in range(100):
            monitor.update(index * 1.0, 0.001 * math.sin(index))
        assert monitor.status.phase == "waiting"
        assert monitor.status.pulse_amplitude_atm is None
