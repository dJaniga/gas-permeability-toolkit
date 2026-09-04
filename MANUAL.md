# gas-permeability-toolkit — manual

The reasoning behind the tool: the physics it implements, the metrology it
reports, the failure modes it guards against, and why the defaults are what they
are.

For installation, the command surface and a working example, see
**[README.md](README.md)**.

## Contents

- [Steady state is required, not optional](#steady-state-is-required-not-optional)
- [Watching it live](#watching-it-live)
- [A supplied downstream pressure](#a-supplied-downstream-pressure)
- [Choosing a flowmeter](#choosing-a-flowmeter)
- [The Klinkenberg correction](#the-klinkenberg-correction)
- [Uncertainty (ISO/IEC Guide 98-3)](#uncertainty-isoiec-guide-98-3)
- [Low-permeability rock](#low-permeability-rock)
- [Pulse decay](#pulse-decay)
- [`preview`](#preview)
- [`summarize` and its findings](#summarize-and-its-findings)
- [Comparing two campaigns](#comparing-two-campaigns)
- [Reprocessing a stored run](#reprocessing-a-stored-run)
- [Reference](#reference)

---

## Steady state is required, not optional

Darcy's law describes *steady* flow. A permeability computed while the rig is
still pressurising describes the transient, and it will look perfectly
plausible. So `collect` gates its result on a detector that requires, over
several consecutive non-overlapping windows and on every monitored signal:

- **scatter** — coefficient of variation within the window below tolerance;
- **drift** — the fractional change an OLS line predicts across the window
  below tolerance.

The drift criterion is the one that matters. A slowly ramping signal has small
scatter inside any short window and would pass a scatter-only test forever.

The reported permeability is the mean over the detected steady window. A run
that never settles still produces a full CSV and summary, clearly marked **not
representative**, and `klinkenberg` refuses it unless you pass
`--allow-unsteady`.

### Stopping once it has held

`stop_after_steady_s` ends the run after steady state has held that long, so a
pressure step can be left to finish itself:

```yaml
stop_after_steady_s: 120    # null = run until Ctrl+C; 0 = stop on confirmation
```

The clock starts when steady state is **confirmed**, not when the plateau began
— the latter is only known in hindsight, and by then a plateau is already
several windows old. So with the default 3×30 s criteria and a 120 s soak, a run
confirming at 90 s stops at 210 s and reports the mean over the whole 210 s
plateau.

If the rig leaves steady state the clock restarts: a hold that was interrupted
did not last. Pair it with `steady_state.max_wait_s` so a rig that never settles
gives up instead of running forever.

## Watching it live

`--plot` opens a window alongside the console output. Every quantity gets its
**own** panel, stacked and sharing a time axis — pressures are not overlaid,
because the point is watching one signal settle at a time.

```yaml
plot:
  panels: [inlet_pressure, outlet_pressure, flow, temperature, permeability]
  window_s: null          # null = whole run from t0; a number = trailing window
  show_criteria: true     # the steady-state bands and the fitted drift line
  show_last_value: true   # each panel's newest value, in its top-right corner
  redraw_interval_s: 0.5  # honoured as set while the plot has its own thread
  max_points: 3600        # per series; the from-t0 view decimates to fit
```

**Every panel prints its newest value in its top-right corner**, in the same
unit as its axis and to the same four significant figures as the console line,
so the plot and the text never disagree about a number you are copying down.
The trace shows the shape and the axis gives the scale, but reading a value off
a plot by eye is guesswork — and it is the number, not the shape, that ends up
in the lab book.

A panel with two traces reads out both, coloured to match, since which of `k`
instantaneous and `k` averaged you are reading matters. A sample with no value
reads `--`, exactly as the console prints it: there is no permeability before
the first computable sample, and none while the flowmeter is unplugged.
Carrying the previous number forward would put a stale value on screen with
nothing to say it was stale, which is the one thing a live readout must never
do. `preview` shows the same readout, following `--volts` — the number you
write down from a wiring check must not claim kPa while the trace is volts.
Set `show_last_value: false` to turn it off.

**The two views answer different questions.** A trailing window fills the axis
with the last few minutes, so small movements are visible while you wait for a
plateau. From t0 shows how far the rig has come since pressure was applied,
which on tight rock is the question that matters. Neither loses data: the from-t0
view decimates to `max_points` with a stride that doubles as the run grows, so a
multi-hour run spans the whole axis at fixed memory rather than silently starting
late.

**Each monitored panel carries the detector's own criteria**, so a signal
creeping out of tolerance is visible long before the console says anything: a
solid line at the window mean, dashed lines at
`mean × (1 ± relative_stddev_tolerance)`, a dotted segment showing the OLS line
the drift criterion is computed from, and the current scatter and drift in the
corner — green passing, amber not. Once a run settles the band is often tens of
times wider than the signal, and letting it set the y-axis would flatten the
trace and hide the drift, so it is left off-scale and the corner says
`(band off-scale)`. The numbers are always shown.

Panels the detector does not watch say **"not a steady-state signal"** rather
than just having no lines: the outlet transducer is never a criterion, and
`steady_state.signals` leaves temperature out by default. The permeability panel
draws the instantaneous value (faint — what the detector tests) over the rolling
mean (bold — what the console reports), and the confirmed steady stretch is
shaded green on every panel.

**The panel set follows the method.** A pulse-decay run drops the flow panel and
gains `delta_pressure` and `decay_fraction`, the latter on a log axis where an
exponential decay straightens into a line and a leak or thermal ramp does not.
No criterion annotations appear there — no detector runs in pulse mode, so
drawing them would assert something never tested.

**With a window open, the run moves to a second thread.** A redraw of a
five-panel figure costs on the order of 0.15 s, against a 0.1 s sample slot at
10 Hz, so drawing inside the per-sample callback spends a large part of the run
not sampling. `collect` and `preview` therefore split the two: the
**acquisition loop runs on a worker thread and the plot keeps the main
thread**.

That order surprises people, and it is not a choice. matplotlib's GUI backends
may only draw on the thread that owns them, which is the main thread — so
moving the *plot* to a worker, the arrangement most people reach for first, is
the one that is actually forbidden. Moving the acquisition instead is what is
left, and it is also what you want: the loop spends nearly all of its slot
asleep, and both the sleep and the DAQ read release the GIL, so the drawing
gets the processor exactly when the loop has no use for it.

Measured on a 12 s run at 10 Hz with five panels, drawing inline against
drawing on its own thread:

| | inline | threaded |
|---|---|---|
| slots starting late | 7 % | **0 %** |
| worst gap between samples | 210 ms | **113 ms** |

With no `--plot` there is nothing to draw and the loop stays where it is; that
is the only difference between the two paths.

Ctrl+C still stops a run, and still stops it cleanly — the first one asks the
loop to finish its current sample, so a partial run is written and summarised
rather than lost. A second interrupt gives up waiting, which matters only if
the DAQ has wedged. Closing the plot window mid-run is not a fault and does not
stop anything.

**The run still reports its own cadence**, because that is the instrument that
tells you whether any of this is working. The loop sleeps to an absolute
target, so it wins back time a slow frame cost by taking the next samples with
no sleep at all — meaning a starved run can hold its mean rate and lose only
its *spacing*, which is what a stuttering console and plot are. The summary
separates the two: `could not hold the configured sample rate` when the rate
genuinely fell short, and `took it in bursts` when only the spacing went. Both
leave the result correct — every reading carries its true elapsed time from the
clock, never its slot index — and both mean fewer or less evenly spaced points
than were ordered.

Note that `max_points` is a **memory** bound, not a speed one. A redraw costs
about the same whether it draws 600 points or 3600, because the time goes on
the axes and their tick labels rather than on the trace; turning it down to
make the plot faster is the natural first thing to try and does nothing.

If the window is closed mid-run, or no display is available, the plot disables
itself and the run carries on.

### Which monitor it opens on

A rig bench usually has the console on one screen and the live plot left running
for hours on the other. Set it once and every plot window follows — `collect`,
`preview`, and the interactive `klinkenberg` and `compare` figures:

```yaml
plot:
  monitor: 2          # 1-based; null = wherever the desktop puts it
  window: fullscreen  # normal | maximised | fullscreen
```

`maximised` fills the work area and keeps the title bar, which is usually what
you want for a window you may need to close; `fullscreen` covers the monitor
including the taskbar and drops the frame.

Placement is **best-effort in every direction**, because a plot is additive to a
run and placement is additive to the plot. An unplugged monitor, a backend
without geometry control, or a failed OS query all leave the window where it
would have opened anyway. The one thing that is never silent is asking for a
screen that is not there:

```
WARNING plot.monitor is 2 but this machine reports 1 screen(s)
        (screen 1: 1920x1080 at (0,0) (primary)). Using screen 1 instead --
        plug the second monitor in, or set plot.monitor: null.
```

Monitors are enumerated through the Windows API rather than the toolkit, because
Tk — the backend matplotlib picks by default here — cannot see individual
monitors at all and reports one merged desktop. On other platforms the query is
not implemented and placement is skipped rather than guessed at.

**On mixed-DPI desktops** a caveat applies. Python with Tk is normally a
DPI-unaware process, so both the OS query and the toolkit report *virtualised*
coordinates. That is harmless while every monitor runs at the same scaling — the
two agree and placement is self-consistent — but a 150 %-scaled laptop panel
beside a 100 % external screen makes the virtual rectangles disagree with the
physical ones, and the window can land short. Making the process DPI-aware would
fix the coordinates and shrink every plot's text, so it is deliberately not done;
matching the scaling of the two monitors is the cheaper fix.

## A supplied downstream pressure

P2 is the outlet transducer by default. When the outlet vents to atmosphere and
that transducer reads noise around zero — or is not fitted — supply the value
instead:

```yaml
downstream_pressure: 101.8        # measured | a number
downstream_pressure_unit: kPa
```

The outlet transducer is **still recorded** in every reading, and the run summary
compares it against the value you declared:

```
! The supplied downstream pressure (101.8 kPa) disagrees with the outlet
  transducer, which read 4.003 kPa over the same window (96.1%). Either the
  declared value is wrong, or the outlet is not actually open to it.
```

That check is the point of keeping both numbers: declaring a pressure while a
valve is quietly shut would otherwise scale every permeability with nothing to
show for it.

P2 sets the apparent permeability *and* the mean pressure, which is the
Klinkenberg regression's own x-axis, so `klinkenberg` refuses to mix runs that
obtained it differently. In the budget a supplied value carries its own
`downstream_pressure_uncertainty` rather than the transducer's specification, and
the P1/P2 covariance term is dropped: a stated constant shares no calibration
error with the inlet.

## Choosing a flowmeter

A rig usually has more than one meter wired — a low-range and a high-range — and
which one suits a given pressure step is an *experiment* decision, not a change
to the bench. So every meter is defined once in `hardware.yaml`:

```yaml
flowmeters:
  low_range:
    channel: ai2
    flow_max: 500.0
    unit: sccm
  high_range:
    channel: ai3
    flow_max: 5000.0
    unit: sccm
default_flowmeter: low_range
```

and each run picks one, in `run.yaml` (`flowmeter: high_range`) or on the command
line. Only the selected meter's analog input is added to the DAQ task; the others
are never read. The meter used is recorded in every run's metadata, because two
runs on the same plug routinely differ in nothing else.

Sizing the meter to the flow is the single most important choice for tight rock —
see [The flow is below the meter's resolution](#the-flow-is-below-the-meters-resolution).

## The Klinkenberg correction

Apparent gas permeability depends on mean pore pressure through slippage;
extrapolating to infinite pressure recovers the liquid-equivalent value. The fit
is linear in `1/P̄`:

```
k_g = k_L + (k_L b) / P_mean
```

`--sample` finds every run recorded for that plug and reports what it found
first:

```
Found 4 runs for core-041 in runs
  core-041_20260804T091200Z        2026-08-04 09:12Z
  core-041_20260804T095100Z        2026-08-04 09:51Z
  core-041_20260804T102300Z        2026-08-04 10:23Z
  core-041_20260804T110400Z        2026-08-04 11:04Z   skipped: never reached steady state
1 run skipped. Pass --allow-unsteady to include them.
```

A run that never settled is skipped with its reason rather than failing the whole
regression — a plug's history legitimately includes aborted attempts.

Three refusals, each because the mistake would otherwise be silent:

- **more than one plug** in a regression. With a directory full of runs from a
  dozen plugs, regressing across rocks is an easy mistake to make.
- **mixed P2 conventions**, since P2 determines the mean pressure this regression
  plots against.
- **mixed methods** — steady-state `k_g` is averaged over a large P1→P2 span
  while pulse-decay `k_g` is at essentially a single pressure, so a systematic
  offset between the methods would masquerade as slippage and land in `b`.

Leak tests are excluded from discovery: a leak is a property of the bench, not a
point on any plug's curve. A [re-derived run](#reprocessing-a-stored-run)
supersedes the original it came from, so one measurement is never regressed
twice.

Results are written per plug — `runs/klinkenberg_core-041.yaml` and its `.png` —
so measuring the next plug never overwrites the last. How many points to take is
your call; nothing nags about it, and after each run `collect` simply says how
many that plug now has.

## Uncertainty (ISO/IEC Guide 98-3)

The measurand is

```
k = 2 Q P_ref mu L / (A (P1^2 - P2^2)),   A = pi d^2 / 4
```

a product of powers, so each input's relative sensitivity coefficient is simply
its exponent: `+1` for flow, reference pressure, viscosity and length; `-2` for
diameter (area goes as `d^2`, which is why the caliper term is usually the
largest in the budget); and for the pressures

```
c_P1 = -2 P1^2 / (P1^2 - P2^2)      c_P2 = +2 P2^2 / (P1^2 - P2^2)
```

which diverge as the differential closes — the quantitative statement of why a
low-differential measurement is a bad measurement.

Type B terms come from the instrument specifications in `hardware.yaml` (a
specification limit divided by its distribution factor: `sqrt(3)` rectangular,
`sqrt(6)` triangular, or the stated `k`); the Type A term is the standard
deviation of the mean over the steady window. `pressure_calibration.correlation`
carries the covariance between the two transducers — because P1 and P2 enter
with opposite signs, positive correlation *reduces* the combined uncertainty.
The coverage factor comes from Student-t at the Welch–Satterthwaite effective
degrees of freedom.

`collect` prints the whole budget ranked by contribution, so it is obvious which
term to improve next. Pulse decay has its own budget with
[no flow term at all](#sizing-the-transducers).

## Low-permeability rock

Below roughly ten microdarcy this rig stops measuring the rock and starts
measuring itself. The symptom is a Klinkenberg fit with a **negative** `k_L` and
a negative slippage factor at a respectable R², from runs that every steady-state
check passed. Nothing is broken; the flow is simply too small for the meter.

### The flow is below the meter's resolution

A 1 µD plug (38.1 mm × 50 mm, nitrogen, venting to atmosphere) passes:

| inlet | flow | on a 500 sccm meter | `u(Q)/Q` at ±0.5 % FS |
|---|---|---|---|
| 5 atm | 0.09 sccm | 0.018 % FS | 1600 % |
| 10 atm | 0.38 sccm | 0.076 % FS | 380 % |
| 30 atm | 3.45 sccm | 0.69 % FS | 42 % |

A thermal meter down there reports its own zero offset. That offset is genuinely
stable, so the detector confirms steady state and is right to — the quantity it
confirmed just is not the sample. A constant `Q0` makes `k_g ∝ 1/(P̄(P̄−1))`,
which is convex in `1/P̄` rather than the straight line Klinkenberg assumes, and
a straight fit through it lands on a negative intercept. A 0.25 sccm offset and
no sample flow at all reproduces the field symptom exactly: `k_L = −0.78 µD`,
`b = −12.7 atm`, `R² = 0.94`.

**Size the meter to the flow.** For a relative flow uncertainty of `target` at a
flow `Q`, a `±0.5 %` full-scale meter needs

```
FS <= Q * target * sqrt(3) / 0.005
```

At 30 atm and 1 µD (3.45 sccm) that is ≤ 60 sccm for 5 % flow accuracy, or
≤ 240 sccm for 20 %. The shipped 500 sccm meter cannot do better than 42 % there.

This is why `flowmeters.*.uncertainty.kind` defaults to `percent_full_scale`:
thermal meters are specified that way, and declaring `percent_reading` instead
understates the flow term by around seventy times — enough to make a
non-measurement look precise. Change it only if your datasheet really says so.

### Equilibration takes hours, not seconds

Pressure diffuses through a plug on a timescale

```
t ~ phi mu L^2 / (k P_mean)
```

At 1 µD and 10 % porosity that is **2.2 h** at 5.5 atm mean pressure and 48 min
at 15.5 atm, against shipped criteria that can confirm a plateau in 90 s. Both
are correct at once: the signal is flat because the core is still filling at a
steady rate. Note the `1/P_mean` — raising the pore pressure shortens
equilibration proportionally, which is the cheapest lever available.

Recording `porosity_fraction` in the sample file lets the tool check this
properly; without it the check falls back to a 5 % porosity as a lower bound and
says so.

### What the tool does about it

Three guards, all after the fact, all reported in the run summary or the fit:

- **A dominated budget.** Any input whose relative contribution exceeds
  `run.uncertainty.max_component_contribution` (default 0.25) is named, and for
  the flowmeter the message gives where the meter sits on its own scale.
- **Too short a run.** `run.steady_state.equilibration_factor` (default 1.0)
  compares the elapsed time against `t` above.
- **A meter stuck at its offset.** `klinkenberg` warns when flow varies by less
  than 5 % across a series whose mean pressure spans more than 2×. This is what
  makes already-recorded runs self-diagnosing: re-run `gasperm klinkenberg
  --sample <id>` and it will say so.

### The real fix is a different method

Those guards tell you the steady-state measurement is not a measurement. They
cannot make it one — below about 10 µD no flowmeter sized for a normal plug can
resolve the flow. **Use [pulse decay](#pulse-decay) instead**, which measures no
flow at all.

One caveat that remains in steady-state mode: the outlet vents to atmosphere, so
`P̄ ≈ P1/2` — mean pressure cannot be varied independently of the differential
without a back-pressure regulator. Pulse decay does not have this problem, since
both vessels sit at the same mean pressure.

## Pulse decay

A core plug between two **closed** vessels, upstream `V1` and downstream `V2`,
both at pore pressure `P̄`. A small pulse `dP0` is applied to `V1`; the
differential decays through the plug, and permeability comes from the decay
*rate*. **No flow is measured**, which is what makes it work at a microdarcy.

Set `method: pulse_decay` in `run.yaml` to make it the default for a rig. It
requires `downstream_pressure: measured` — a declared constant P2 asserts the
outlet is open to something, which contradicts a closed vessel, and `collect`
refuses the combination at config load rather than three minutes in.

### The physics

Two models, both implemented and both exact in the package's CGS-Darcy units —
`k` in darcy, `A` in cm², `µ` in cP, `L` in cm, `V` in cm³ and gas
compressibility in **1/atm** give `alpha` in 1/s with no conversion constant at
all.

**Zero storage (Brace et al. 1968)**, when the plug's pore volume is small
against the vessels:

```
alpha = k*A / (mu*c_g*L) * (1/V1 + 1/V2)
```

**Sample storage (Dicker & Smits 1988)**, when it is not — the usual case once
the vessels are small enough to give a workable run time. With `V_p = phi*A*L`,
`a1 = V_p/V1`, `a2 = V_p/V2` and `theta_1` the first root of the storage
equation:

```
alpha = theta_1^2 * k / (phi*mu*c_g*L^2)
```

Applied automatically whenever `sample.porosity` is recorded
(`storage_correction: auto`). Without it the zero-storage form reads **low** — by
1.8 % on the shipped 400/75 cm³ vessels, and by 20 % on 5 cm³ ones.

Porosity is the one place a pulse-decay run takes a *petrophysical* number as a
measurement input rather than as provenance, so how it is written matters.
`porosity_unit` accepts `fraction` or `%` (with `v/v`, `pct` and `p.u.` as
aliases), and `porosity_uncertainty` is in the **same** unit — 0.5 against a
percentage is half a percentage point, not half a percent of the reading. A
percentage left labelled `fraction` is refused at load: it would otherwise put a
porosity of 1040 % into the storage correction and read the permeability wildly
low. The older field name `porosity_fraction` still loads, and still means a
fraction.

### How long a run takes

`k` and `tau` are inversely proportional, and `tau` also scales as `1/P̄`. For a
38.1 × 50 mm plug at 10 % porosity in nitrogen at 10 atm, on 400/75 cm³ vessels:

| k | time constant | run to dP/dP₀ = 0.4 |
|---|---|---|
| 1 µD | 13.9 h | 12.8 h |
| 3 µD | 4.6 h | 4.3 h |
| 10 µD | 1.4 h | 1.3 h |

`collect` prints this prediction at startup when
`pulse_decay.expected_permeability` is set — nobody should discover a
fourteen-hour run by starting one. Two levers shorten it: **charge to a higher
pore pressure** (exactly proportional), and **shrink the vessels**. Note the
decay rate is set by `V1·V2/(V1+V2)`, which the *smaller* vessel dominates —
enlarging only the big one changes almost nothing.

**Running the decay to 5 % is wasted time, and slightly worse.** Simulated fits
at 1 µD, treating the full 0.25 % FS accuracy spec as scatter:

| stop at dP/dP₀ | run length | σ(α)/α | bias |
|---|---|---|---|
| 0.9 | 1.5 h | 7.4 % | +2.1 % |
| 0.7 | 5.0 h | 2.1 % | +0.1 % |
| **0.5** | **9.7 h** | **0.9 %** | −0.1 % |
| 0.3 | 16.8 h | 0.5 % | +0.1 % |
| 0.05 | 41.8 h | 0.7 % | +3.2 % |

Late samples are noise-dominated, so they add scatter rather than information.
Hence the shipped fit window of 0.90 → 0.50 and a stop at 0.40, not the textbook
0.05.

### Configuration

The vessels and the transducers are bench hardware, so they live in
`hardware.yaml`:

```yaml
reservoirs:
  upstream:
    vessel: 380.0     # the calibrated vessel itself
    dead: 20.0        # tubing + ports + valve internals to the plug face
    unit: cm3
    method: gas expansion
  downstream:
    vessel: 65.0
    dead: 10.0
    unit: cm3
  spacer_types:       # the bores this rig owns; lengths are per-run
    wide:
      internal_diameter: 25.4
      dimension_unit: mm
    narrow:
      internal_diameter: 12.7
      dimension_unit: mm
  correlation: 0.0    # both sensitivities are POSITIVE, unlike P1/P2

pulse_transducers:    # null reuses the inlet/outlet pair
  upstream:
    channel: ai4
    volts_max: 10.0
    value_max: 100.0
    unit: bar
  downstream:
    channel: ai5
    volts_max: 10.0
    value_max: 100.0
    unit: bar
```

**Dead volume, not nameplate volume.** `vessel` and `dead` are separate because
they are established by different means and change independently — the vessel is
a calibrated object, while the dead volume is the tubing, transducer port and
valve internals up to the plug face, which changes whenever the rig is
replumbed. Both go into V1 identically, and the dead volume is the one routinely
forgotten. Measure both by gas expansion against a reference vessel, not from a
drawing.

The run-level knobs live in `run.yaml`:

```yaml
method: pulse_decay
pulse_decay:
  min_pulse_pressure: 20.0
  pulse_pressure_unit: kPa
  max_pulse_fraction: 0.10     # largest dP0/P_mean the linearisation allows
  stop_below_fraction: 0.40    # end the run here
  fit_start_fraction: 0.90     # fit from here ...
  fit_end_fraction: 0.50       # ... down to here
  fit_bin_s: 1.0               # bin before fitting; null = every sample
  fit_offset: true
  storage_correction: auto     # auto | brace | dicker_smits
  expected_permeability: 1.0
  expected_permeability_unit: uD
```

### Spacers

A hollow spacer fitted upstream of the core face is part of the flow path, so
its internal volume adds to V1.

A spacer is characterised by **two** measurements, established in different
places. The **internal diameter** belongs to a set of parts bored to one size, so
it is bench hardware and lives in `spacer_types`. The **length** differs from
spacer to spacer even within a type, so it is given per fitted spacer at run time
— the stack is made up to suit the plug in the holder and changes between runs
without the bench changing. Same split as the flowmeters: defined once, selected
per run.

```bash
gasperm collect --sample samples/core-041.yaml --method pulse_decay \
    --spacer wide:50 --spacer wide:25 --spacer narrow:30
```

Repeat `--spacer` to stack; the length is in the type's own `dimension_unit`.
`--spacer none` declares an empty holder, which is how you override a stack that
`run.yaml` sets by default:

```yaml
pulse_decay:
  upstream_spacers:
    - {type: wide, length: 50.0}
    - {type: narrow, length: 30.0}
```

Volume is the cylinder `π(d/2)²·L` plus each type's `end_correction_cm3`, which
covers chamfers, o-ring grooves and counterbores the plain cylinder misses.
**Bore enters squared**, so its caliper uncertainty counts double — the same
reason the plug's diameter dominates the steady-state budget. An unknown type
name is fatal at startup and names the bores you do have, so a typo never becomes
a quietly wrong V1.

How much a miscounted stack costs you **depends on the rig**, so `collect` prints
the figure at startup rather than asserting a rule. What scales the permeability
is the *effective* volume `V1·V2/(V1+V2)`, which the **smaller** side dominates:

| | 3 × 5 cm³ spacers shift k by |
|---|---|
| V1 = 400, V2 = 75 cm³ | **0.6 %** |
| V1 = V2 = 20 cm³ | **27 %** |

So on a large upstream vessel the stack barely matters — but it matters a great
deal once you shrink the vessels to get the run time down, which is exactly the
change you would make for a microdarcy plug. The stack is recorded in every run's
summary as `type:length` entries, so a series measured at different stack heights
is never confused with one measured at the same.

Their uncertainty follows how the two measurements actually correlate. **Bore
error is shared** by every spacer of a type — they are bored to one spec — so it
sums within a type and adds in quadrature across types. **Length error is
independent**, since each spacer is measured separately, so those grow as
`sqrt(n)`. Treating the bore as independent too would understate the stack by
roughly `sqrt(n)`.

### Sizing the transducers

This is the decisive question, and it is the same trap as the flowmeter wearing
different clothes. At 10 atm mean with a 101 kPa pulse:

| | u(P) | as a fraction of the pulse |
|---|---|---|
| 0–68.95 MPa at 0.25 % FS | 99.5 kPa | **139 %** |
| 0–100 bar at 0.25 % FS | 14.4 kPa | 20 % |

`collect` warns at startup when the pulse is small against the transducer's
specification. But note **what does and does not matter**: `alpha` is a *rate*,
so a constant gain error leaves it unchanged and the fitted offset absorbs a
constant zero error exactly. Transducer gain and zero — most of a datasheet
accuracy figure — cancel out of this measurement. What does not cancel is their
**noise**, which sets the scatter of the fit and appears directly as the Type A
`u(alpha)` in the budget. A low-range or differential pair is the fix.

### Reading the result

```
  method              pulse decay -- Dicker & Smits model
  pulse               dP0 = 50.73 kPa at t = 1.0 s   (5.01% of P_mean)
  decay fit           ACCEPTED   1.5-4.5 s, 590 pts
                      alpha = 1.9991e-01 1/s +/- 0.31%,  tau = 5 s
                      R^2 = 1.000000,  offset = +0.08093 kPa,  rho_1 = 0.11
  vessels             V1 = 8 cm3, V2 = 8 cm3    a1 = 0.713, a2 = 0.713
                      theta_1 = 1.1273   (the zero-storage form would read 10.8% low)
```

The fitted **offset** is the two transducers' zero mismatch; leaving it out of
the model biases `alpha` low, by 5 % for a 0.5 kPa offset on a 50 kPa pulse and
by 33 % for 5 kPa. `rho_1` is the lag-1 residual autocorrelation: consecutive DAQ
samples are not independent, so the decay is binned before fitting and a `rho_1`
still above `max_residual_autocorrelation` means `u(alpha)` is understated.

Every run also writes `decay_fit.png` — the decay on a log axis with the fit over
it, and the residuals below, where a leak or a thermal ramp shows as structure
long before it shows in R².

Pulse-decay runs feed [`klinkenberg`](#the-klinkenberg-correction) exactly like
steady-state ones, one run per mean pressure.

### The pre-step: a leak test

**Do this before any measurement.** A leak produces a differential decay
indistinguishable from a slow sample, so without a bound on it a pulse-decay
result could be entirely the apparatus. It is the pulse-decay counterpart of
checking the flowmeter's zero with the inlet closed.

Blank or bypass the plug, charge both vessels to the pressure you will measure
at, apply the same pulse, and run:

```bash
gasperm collect --sample samples/core-041.yaml --leak-test
gasperm collect --sample samples/core-041.yaml --leak-test --plot
```

`--leak-test` implies `--method pulse_decay`. Unlike a measurement this is a
**fixed observation, not a decay to be waited out** — on a tight rig the ideal
outcome is that nothing happens, so there is no completion signal to stop on and
it runs for `pulse_decay.leak_test_duration_s` (default 1 h). A run without a
duration is refused rather than left to run forever.

The result is reported as **the permeability the apparatus alone would fake**,
which is the number your sample has to stand clear of:

```
  purpose             LEAK TEST -- the apparatus, not the sample
  leak equivalent     0.031 uD +/- 0.006 uD
  ! LEAK TEST: the blanked rig decays at 8.4e-06 1/s, which is what a sample of
    0.031 uD would look like. At pulse_decay.max_leak_fraction (5%) that puts the
    floor for a trustworthy measurement at about 0.62 uD.
```

and a tight rig reports the outcome you want:

```
  leak                NONE MEASURABLE -- the blanked rig held its differential
```

An hour of watching a differential is where `--plot` earns its place. The window
is titled `LEAK TEST (the apparatus, not the sample)` and the permeability panel
is labelled `leak equiv.`, so a plot left on screen is never mistaken for a
measurement. The `dP/dP0` panel is pinned to the span between
`stop_below_fraction` and 1 rather than autoscaled: on a rig that is holding the
differential is constant to a few parts in 10^5, and a log axis fitted to *that*
would render the flat line you want to see as violent noise. A tight rig reads as
a line along the top; a leak curves away from it long before the fit confirms it.

**Every later run compares itself against the most recent test automatically** —
found by rig, not by plug, since the apparatus leaked the same whichever core was
in it:

```
  leak test           core-041_20260810T125426Z: 4.28e-02 1/s = 150 uD  (30.0% of this decay)
  ! The rig's own decay is 30.0% of the one measured here, above
    pulse_decay.max_leak_fraction (5%) ... at this ratio you are largely
    measuring the apparatus.
```

It also warns when **no** leak test has been done at all, and when the one it
found was at a materially different pore pressure — leak conductance depends on
pressure, so a test that passed at 3 atm says nothing about 30 atm.

**Subtraction is off by default.** Setting `pulse_decay.leak_correction:
subtract` takes the leak rate off the measured one, which is defensible for a
linear, stable leak — the leak path is in parallel with the plug, so the rates
add. But a leak that changed since the test would move the result with nothing to
show for it, so the shipped behaviour is to compare and warn, and the correction
says loudly when it has been applied.

### Two more checks, and one thermal trap

Three pulses at one `P̄` should agree within their combined `u(alpha)` — a
monotone walk means the plug is still equilibrating from the previous step. And
the observed `tau` should match the one predicted from the recovered `k`: much
shorter means a leak, much longer a blocked line or a closed valve.

A 0.1 K room swing moves a closed vessel by `dP/P = dT/T` — 0.34 kPa at 10 atm,
drifting over hours, which looks exactly like a slow exponential. The fitted
offset absorbs a constant thermal bias but not a ramp, so `collect` compares the
temperature drift across the fit window against the fit's own residuals and says
so when they are comparable.

## `preview`

A diagnostic view for checking that a transducer reads what you think it does,
and how noisy it is right now — with no plug in the holder and nothing being
measured.

```
     time       P1 (kPa)       P2 (kPa)  Q:low_range (sccm)        ai7 (V)
     4.5s      1.013e+04           101.3               0.412          3.301
```

**It computes nothing and stores nothing** — no permeability, no gas lookup, no
run directory, no CSV. That is what makes it a signal check rather than a
measurement: a command that also derived a permeability would have to invent a
sample, a gas and a geometry to do it. It never asks for a sample file either; it
describes the **bench**, reading `hardware.yaml` and `run.yaml` and stopping
there.

**Only the channels you select are opened**, which is what lets you watch the
flowmeter a run is *not* using (`-s flow.high_range`), or a bare input with no
calibration at all (`-s ai7`, raw volts over the widest range the 6421 supports),
without editing a config file.

Signals are shown in their **configured display unit**
(`run.display_pressure_unit`, `display_flow_unit`, °C), overridable per signal as
`NAME:UNIT`. Pressures are shown **absolute** — the same number `collect` would
compute from the same voltage — with the banner saying so when the transducer is
a gauge type. For checking a zero, `--volts` is the better tool: it shows the
reading before any calibration has an opinion on it.

**`-s pulse` picks the pulse-decay transducers automatically**, resolving the
pair exactly as a pulse-decay run does: the dedicated low-range pair when
`hardware.pulse_transducers` defines one, the steady-state inlet/outlet pair when
it does not. Either way the banner says which you got:

```
  pulse_upstream        kPa   ai4  0-10 V -> 0-100 bar (absolute)  [dedicated pulse transducer]
  pulse_upstream        kPa   ai0  0-5 V -> 0-68.95 MPa (absolute)  [NO dedicated pulse pair -- falls back to the steady-state transducer]
```

A pulse pair that silently turned out to be the 0–68.95 MPa transducers is the
failure the whole method exists to avoid — they cannot resolve a 100 kPa pulse
(see [Sizing the transducers](#sizing-the-transducers)) and the run would look
healthy while measuring nothing. `--list` warns about it up front. Unlike
`collect`, this ignores `run.method`: you check the dedicated pair on a rig whose
`run.yaml` still says `steady_state`, usually because you are about to change it.
`pressure` is a pair the same way, and a unit on a group applies to every member.

**`-s pulse` also brings `pulse_dp`**, the difference of the two — which is what
pulse decay actually measures, and what neither transducer panel shows. A 100 kPa
pulse riding on 50 bar of pore pressure is a fifth of a percent of either
absolute trace, invisible on an axis scaled to 5000 kPa; on its own panel it is
the full height of the plot:

```
     time      pP1 (kPa)      pP2 (kPa)       dP (kPa)
     0.0s           5020           5000             20
     0.2s           5015           5000             15
```

That panel is where the two things worth knowing before a fourteen-hour run
show up. The **zero mismatch** between the pair — with both vessels open to the
same pressure, dP should read zero and generally does not — is exactly what the
free-offset fit exists to absorb (see [Reading the result](#reading-the-result));
seeing its size beforehand tells you whether it is a few tenths of a kPa or
enough to matter. And the **noise floor on the difference** is what the decay
has to be resolved against, which on a tight plug is the thing that decides
whether the run is worth starting.

The subtraction is done on the two converted **absolute** pressures, so a gauge
pair's atmospheric term cancels rather than being counted twice. `pulse_dp` is
the one signal that keeps its pressure unit under `--volts`: no wire carries a
difference, and a difference of two voltages is a pressure only if both
transducers share a span, which the fallback pair need not. Its two members still
show their own volts. It is selectable on its own (`-s pulse_dp`), including on a
rig with no dedicated pair, where it is the steady-state differential.

With `--plot`, each signal gets its own stacked panel. **Nothing is drawn on top
of the traces** — preview runs no detector, so any criterion band would assert
something never tested. The DAQ is sampled at `daq.sample_rate_hz` (or `--rate`)
so the plot sees the real noise; the console refreshes at 2 Hz in place, and the
final sample is always printed so a preview never ends on a stale line.

The temperature probe is opened **only** if `temperature` is among the selected
signals. Asked for explicitly and failing is fatal; merely part of the default
set and failing drops its column with a warning, and the DAQ half carries on.

## `summarize` and its findings

A plug accumulates history — several pressure steps, a leak test, an aborted run,
a re-derivation after a calibration was corrected, and for an exposure study the
whole lot again months later. That is the right way to *store* it and a poor way
to *read* it.

```
core-041
  5.000 x 3.810 cm   porosity 10.1432 +/- 0.25 % (helium pycnometry)   bulk density 2.36145 g/cm3
  8 confirmed run(s), 1 not   2026-01-10 to 2026-06-14

  Runs   pressures in kPa, permeability in mD;  pulse rows: P at the pulse
    run                      date        method            P_in     P_out    P_mean       dP0         k      U(k)  meter
    core-041_20260110T090000 2026-01-10  steady_state    759.94    253.31    506.62                 0.9   0.03776  low_range
    ...
    core-041_20260114T090000 2026-01-14  pulse_decay     3090.4    3039.8    3039.8    50.663    0.0144 0.0006041  low_range
    core-041_20260111T000000 2026-01-11  steady_state    1823.9    608.00    1215.9                 0.9   0.03776  low_range   never confirmed

  Klinkenberg correction
    k_L = 0.520566 mD +/- 0.02125 (k = 2.45)
    b   = 4 atm    R^2 = 0.9373    8 points   weighted

  Findings
    - These runs fall into two groups either side of 2026-06-14. ...
    - core-041_20260111T000000Z is not a measurement: never confirmed a measurement.
```

**The petrophysics on the identity line is quoted as the sample file states
it** — unrounded, and in the unit it was entered in. A helium pycnometer
reports percentage points to five or six figures, and this line is read against
the file it came from; restating 10.1432 % as a four-figure fraction makes the
two disagree, which is exactly what an identity line exists to prevent. So the
porosity carries its `porosity_unit` and, where one is recorded, its
`porosity_uncertainty` in that same unit, and `bulk_density_g_cm3` is shown
beside it — porosity is cross-checked against the densities (`1 - rho_b/rho_g`),
and a page carrying one without the other cannot be checked at all.

The digits stop at twelve significant figures rather than at `repr`, which
suppresses the binary tail a unit conversion leaves behind: a porosity of 10.4 %
is 0.10400000000000001 as a fraction, and printing that would read as spurious
precision rather than as fidelity. A run recorded before the entered value was
kept has only its converted fraction, which *is* its full resolution — it is
printed unrounded, with no unit and no percent sign invented for it. `--output`
writes both spellings under separate keys (`porosity` with `porosity_unit` and
`porosity_uncertainty`, beside `porosity_fraction` and `bulk_density_g_cm3`), so
a parsed summary never has to redo the conversion to check it against the file.

**Pressures are shown in `run.display_pressure_unit`**, the same unit `collect`
printed on the console and the live plot labelled its axes with — not the atm
the physics runs in and the files store. The unit is named once above the table
rather than repeated in three headings, which would push it past the width of a
terminal. `P_mean` is exactly the midpoint of the `P_in` and `P_out` beside it,
so the three read as a set; `P_out` is the P2 the equation *used*, which on a
rig with a declared downstream pressure is the declared number rather than the
transducer.

`dP0` is the pulse a decay started from, and the column appears only when the
plug has at least one pulse-decay run — a steady-state row leaves it blank,
because there is no pulse, which is not the same as a pulse whose amplitude
went missing. A run recorded before the summary carried its pressure pair shows
`--` for `P_in` and `P_out`: they cannot be recovered from a mean, so they are
reported as unknown rather than split evenly and guessed at.

**On a pulse-decay row, `P_in` and `P_out` are the pressures at the pulse** —
the *setup condition*, which is what re-measuring a plug under matching
conditions actually needs. The upstream vessel decays toward the downstream for
the whole run, so its **mean** collapses onto the pore pressure: it is nearly
equal to `P_mean`, nearly equal to the outlet's mean, and is not a number
anyone can set a regulator to. The caption says `pulse rows: P at the pulse`
whenever the table contains one.

So a pulse row reads as one moment: `P_out` is the pore pressure both vessels
were charged to, `P_in` is that plus the pulse, and `dP0` is their difference.
Note that reading `dP0` *by subtracting the two columns* loses precision — at
3000 kPa, five significant figures leaves ±0.05 on each, so a 50 kPa pulse
comes out uncertain in its second digit. That is why `dP0` has its own column
rather than being left as an exercise. The full-precision values are in the
run's own sidecar as `pulse_decay.initial_upstream_pressure_atm` and
`initial_downstream_pressure_atm`, and in `--output` as
`initial_inlet_pressure_atm` / `initial_downstream_pressure_atm` alongside the
window means.

A steady-state row is unaffected: it keeps the means over its measured window,
which for a run held at a fixed differential is the same thing throughout.

**The findings are the point; the table is the evidence.** A summary that only
restated what is on disk would leave you to notice that a run never confirmed,
that two meters were used where one should have been, that a series is one
pressure short of a fit, or that a pulse-decay campaign has no leak test behind
it. Each is reported with what it means for the result.

It also **notices when the history is two campaigns rather than one**. Runs
cluster in time — a day of pressure steps, a month of nothing, another day — and
when that gap is unmistakable the summary names the date and points at
[`compare --split`](#comparing-two-campaigns), because a plug measured either
side of a treatment is a paired experiment whose result is the *difference*. A
fit spanning both is regressing two states as one, which is usually what a poor
R² is telling you. The thresholds are deliberately conservative — several times
the plug's own typical spacing **and** at least three days, with two runs either
side — so pressure steps hours apart never read as two campaigns, and neither
does monthly monitoring.

## Comparing two campaigns

### The measurand is the change, not either value

This is the whole reason it is a command rather than a mental subtraction.
Errors **common to both measurements** move both results the same way and are
absent from their ratio: the same plug's geometry, the same transducer on the
same calibration, the same flowmeter, the same viscosity model. What survives is
the scatter — usually far smaller.

So a rig reporting `U(k) = 20 %` on each of two runs can still resolve a 5 %
change between them. The report says so explicitly:

```
    k_L   liquid-equivalent permeability
        0.5 -> 0.545 mD   (delta +0.045)
        increased 9.00% +/- 3.53% -- SIGNIFICANT
        u_c = 1.27%, k = 2.78 (v_eff = 4.0); smallest change this could resolve: 3.53%
        For reference the ABSOLUTE uncertainties are 2.14% and 2.14%; the ratio is
        better determined than either because the shared inputs cancel.
```

Formally, for a ratio `R = k_B/k_A` built from the same inputs (GUM 5.2):

```
u_rel²(R) = Σ [ c_i,A·u_i,A − c_i,B·u_i,B ]²          over SHARED inputs
          + Σ [ (c_j,A·u_j,A)² + (c_j,B·u_j,B)² ]     over INDEPENDENT inputs
```

Note what the shared term does when the two readings differ: it is a
*difference* of contributions, not zero. Two runs at 10.0 and 10.2 atm on a
percent-of-full-scale transducer share an absolute error, so the formula charges
exactly that residue — automatically, with no special case. **Matched conditions
are not a precondition this command asserts; they are a quantity it prices.**

### Every cancellation is itemised

A claim that an uncertainty went away is the load-bearing part of the result, so
it comes with its evidence, in the report and in the `--output` file:

```
  What cancelled between the two measurements
    L        sample length              100.0% removed   same plug, same recorded value
    Q        gas flow rate              100.0% removed   same instrument/model (flowmeter specification)
    mu       gas viscosity              100.0% removed   same instrument/model (coolprop viscosity)

  What did not, and therefore sets the detection limit
    rep      repeatability                1.13% of the ratio   Type A -- an independent draw each run
```

Three rules decide sharing, per component rather than by one global switch:

- **Type A never cancels.** Scatter is an independent draw each run, however
  alike the runs were. It is what sets the detection limit.
- **Plug inputs** (`L`, `d`, `phi`) cancel only for the same plug *and* only
  while the recorded value is unchanged. If the plug was measured again between
  campaigns, those are two independent caliper readings and the cancellation is
  void — detected from the values themselves, not from an assertion, and said
  out loud.
- **Rig inputs** cancel even between *different* plugs, because both were
  measured on the same bench with the same instruments.

### What it refuses, and what it reports

A different method, gas, or P2 convention between the two campaigns is
**blocking**: the difference would be that mismatch plus whatever the sample did,
and nothing can separate the two. `--allow-mismatched-conditions` reports it
anyway, still flagged. A changed flowmeter is survivable but voids the meter's
cancellation, so it is charged to the comparison in full, on both sides.

It reports `k_L` and `b` when both sides have a Klinkenberg fit, apparent `k_g`
for every matched mean pressure, and porosity when both sides recorded it. Two
details worth knowing:

**`b` is a second observable, and often the sharper one.** It depends on pore
throat size relative to the gas mean free path, so it can move before `k` does
when pore structure changes. It is also immune to anything that merely scales the
series, so its uncertainty comes from the two regressions alone.

**A mean-pressure mismatch is quantified rather than hoped away.** Where two
matched runs sat at different pressures, the fitted `b` says how much of the
apparent change that alone accounts for — the difference between "permeability
fell 12 %" and "fell 12 %, of which 3 % is the pressure mismatch".

## Reprocessing a stored run

Every run keeps its **raw voltages** and the raw probe temperature alongside the
derived values, so a measurement can be recomputed without repeating it. A
calibration certificate arrives; a porosity is finally measured with a stated
uncertainty; a plug is re-measured with better calipers. None of that needs a
fourteen-hour pulse-decay run again.

### Three classes of change, and only two move the answer

| class | example | effect |
|---|---|---|
| **result** | geometry, calibration constants, gas, vessel volumes, fit window | `k` itself moves — a **correction** |
| **uncertainty** | `porosity_uncertainty`, any `*.uncertainty` spec, coverage probability | only `U(k)` moves — a **re-costing** |
| **metadata** | operator, notes, lithology, display units | neither moves |

```
Reprocessing 6 run(s) from raw voltages
  uncertainty: moves U(k) only -- k is untouched
    sample.porosity_uncertainty: None -> 0.005

  run                                    k (mD)              U(k) (mD)   verdict
  core-041_20260110T090000Z    0.525982 -> 0.525982     0.1024 -> 0.1149   k unchanged, U re-costed
```

**The prediction is checked against the arithmetic.** If a field predicted
`uncertainty` turns out to move `k`, that is reported loudly rather than trusted
— it means either the classification is wrong or the field is coupled to the
physics in a way nobody noticed. The table is advisory; the recomputation is
authoritative. Conversely, a change that did nothing says why: porosity enters
the budget only through the Dicker–Smits storage correction, so changing its
uncertainty on a *steady-state* run is a legitimate no-op, and saying so is the
difference between that and a typo.

### It never edits the original

Reports only, unless `--write`. With `--write`, each re-derived run goes to a
**new** directory named `<original>_reprocessed`, carrying a copy of the same raw
CSV plus a `derived_from` block naming its parent and every field that changed.
The original is the record of a measurement; silently rewriting one would make
every report already issued from it unreproducible.

```yaml
derived_from:
  run: core-041_20260110T090000Z
  reprocessed_at: '2026-08-17T19:31:42+00:00'
  changes:
    - field: sample.porosity_uncertainty
      before: 0.01
      after: 0.03
      predicted: uncertainty
  permeability_moved: false
  uncertainty_moved: true
```

A derived run **supersedes** its parent everywhere runs are reduced to
measurements — in `klinkenberg`, `compare` and `summarize` alike — so one
experiment is never counted twice.

Reprocessing starts from each run's **own stored config snapshot**, not from
whatever the config files say today, or the "before" half would be a result
nobody ever produced. `--from-config` opts into the current files, for when the
rig file itself is what was corrected.

### A batch is re-derived in parallel

```bash
gasperm reprocess --all --write         # one worker per CPU
gasperm reprocess --all --write -j 4    # four at a time
gasperm reprocess --all --write -j -2   # every core but one
gasperm reprocess --all --write -j 1    # all in this process
```

`-j` follows joblib's convention, since that is the one people arrive with: a
positive number is that many workers, `-1` is every CPU, `-2` every CPU but one.
`-2` is the one to reach for while you still want the machine.

A replay costs what the original acquisition's *arithmetic* cost, and that is
per sample: every reading goes back through the same processor, which looks the
gas properties up at that reading's own temperature and pressure and, for a
pulse run, solves the storage equation for `theta_1`. A fourteen-hour run at
10 Hz is half a million samples. One run is tens of seconds; a season's work on
a bench is an hour.

The property that makes this parallelisable is the same one that makes `--all`
safe: **runs do not interact**. Each re-derives from its own snapshot, reads its
own CSV and builds its own property provider, so there is no shared state to
guard and no ordering to preserve between runs.

Worker **processes**, not threads. The time is spent inside CoolProp's `PropsSI`
and SciPy's `brentq`, neither of which releases the GIL for the scalar calls
made here, so threads would queue up on one core and change nothing.

Three details that are only visible when they are wrong:

- **Results keep the caller's order.** They are reported as a table keyed by
  run, and completion order is whatever the scheduler decided.
- **A run that cannot be replayed is still just a skip.** Its exception comes
  back in its slot rather than being raised, so one unreadable CSV does not cost
  the other forty runs — exactly as it did serially.
- **The longest run is started first.** Handed out last, a fourteen-hour decay
  would still be fitting long after every short burst had finished and the pool
  had gone idle.

Below a few megabytes of raw record the batch stays serial: a worker pays a
fresh interpreter start plus the CoolProp and SciPy imports, a second or two,
and a handful of short runs cannot win that back. An explicit `-j` overrides
that either way — `-j 1` for a readable traceback, `-j N` when you know what the
batch costs.

`-j` is also the memory lever. A worker holds its whole run in memory as
`Reading` objects, roughly twenty times the size of the CSV — an 86 MB record is
about 1.7 GB — and the default runs one per CPU. That is comfortable on a rig
whose runs are minutes long and not on one whose records are tens of megabytes
each. There is no portable way to ask how much memory is free, so the default
sizes itself against CPUs; lower `-j` if the machine starts swapping.

### Re-measured plugs, a whole bench at once

```bash
gasperm reprocess --sample C12 --sample-file samples/C12.yaml   # one plug
gasperm reprocess --all --samples-dir samples --write           # every plug
```

Editing `samples/C12.yaml` on its own changes nothing: a reprocess starts from
each run's own stored snapshot, which is the whole point — otherwise the
"before" half of the comparison would be a result nobody produced. The new file
has to be handed in.

`--sample-file` is one file for one plug, and it is **refused with `--all`**: it
replaces the entire sample section, so across a bench it would stamp one plug's
id, geometry and porosity onto every other core. `--samples-dir` has no such
failure mode — each run says which plug it measured and its file is looked up by
that name, so a batch re-derives every core against *its own* re-measurement.
The id inside the file must still agree with the one the run recorded; a file
named for one plug carrying another's id is a copy-paste, and applying it would
quietly re-label the measurement. That run is refused, and the rest of the batch
carries on.

Whether this moves `k` or only `U(k)` depends on the method. Porosity enters the
Darcy equation not at all, so on a steady-state run a re-measured porosity is a
legitimate no-op — but it is an input to the Dicker–Smits storage correction, so
on a pulse-decay rig it is a **correction**, and the re-derived runs supersede
their parents everywhere runs are reduced.

One reporting detail follows from the per-plug form. With `--set`, every run
moves a field to the same new value, so the summary can name it once. With a
folder, each core moves its own porosity to its own number, and the summary says
`a different value per run` rather than quoting one — the per-run before and
after are in each written run's `derived_from` block.

### Checking that a replay reproduces its original

```bash
gasperm reprocess --verify --all                    # every run in the directory
gasperm reprocess --verify runs/C14_2026...
gasperm reprocess --verify --all --tolerance 1e-5   # only the real outliers
```

A no-change reprocess **must** reproduce the stored result. If it does not, no
reported change can be attributed to the field that was actually edited — the
baseline has moved underneath it. `--verify` re-derives with nothing changed and
compares against what is stored, writing nothing and exiting `2` if any run
fails, so it can gate a script.

The useful output is not the size of the difference but the **stage** it appears
at, because the three stages fail for unrelated reasons:

```
  C14_20260821T075735Z    k    -0.0540%   DOES NOT REPRODUCE
      the per-sample values are exact but the averaged window is not the stored
      one, so the reduction covers different samples
      per-sample k   worst relative drift 3.019e-08
      window         stored 0.05-20.95 s   replayed none
      summary k      0.208702 -> 0.208591 mD
```

- **per-sample** drift means the same voltages are producing different values —
  a calibration or property-lookup problem.
- **window** means the derivation is exact but the two paths disagree about
  *where* the measurement was.
- neither, yet `k` still moved, means the reduction arithmetic itself differs.

Small differences are expected and ignored: a replayed window bound comes back
through four decimals of `elapsed_s`, and a pulse-decay run re-*fits* its
exponential, which reproduces to about a part in 1e7 rather than exactly.

**`--tolerance` sets what counts as reproduction**, as a relative difference.
The default is `1e-6` — loose enough to absorb that refit, tight enough to catch
a real defect. Raise it (`--tolerance 1e-5`) on a rig whose replay is known to
differ slightly and you want only the outliers; it cannot be zero, because a CSV
round trip does not reproduce a float exactly and every run would fail. The
threshold is printed with the verdict:

```
Verifying 8 run(s) re-derive to their stored results   (tolerance 1e-05)
```

A pass says nothing without the threshold it was judged at, and an operator
reading the output months later cannot ask what it was. It applies to the
per-sample drift and to `k` and `U(k)`; the **window** is compared in seconds
against the CSV's stored precision instead, which is a different question and
not one worth tuning.

`k` and `U(k)` are checked separately, because they move independently: a
re-costing bug leaves `k` exactly where it was and moves only the budget, which
a check on `k` alone would wave through.

### Reprocessing the whole bench

`--sample <plug>` re-derives one plug's whole campaign, which is the usual case:
a corrected uncertainty applies to a campaign, not to a single run. **`--all`**
re-derives every run in the runs directory, across every plug — for a *rig*-level
correction, where a recalibrated transducer or a re-measured vessel applies to
everything the bench ever recorded.

```bash
gasperm reprocess --all --from-config           # what would it change?
gasperm reprocess --all --from-config --write   # commit it
```

Each run still re-derives from **its own** snapshot, so every plug keeps its own
geometry and porosity; nothing is broadcast between cores. Because of that, a
`--set` that names a field describing one core (`id`, `length`, `diameter`,
`porosity`) is warned about — applied across a directory it would corrupt every
plug but the one you had in mind, and each result would still look internally
consistent. `--sample-file` is refused outright for the same reason: it replaces
the whole sample section. Fields that describe the *method* rather than the core,
`porosity_uncertainty` among them, pass without comment.

**Runs already superseded by an earlier re-derivation are skipped**, and the
command says which. Re-deriving one would leave its parent with two children,
and since supersession keeps every childless run, that single experiment would
then enter a regression twice — the exact thing the mechanism exists to prevent.
Running `--all --write` twice therefore builds a *chain* per experiment
(`..._reprocessed_reprocessed`), and each reduction still sees one run per
experiment.

One more thing changes with a batch: the per-run changes need not be the same.
Each run diffs against its own snapshot, so a value already correct in one run's
config is not a change there. A change that applied to only part of the batch is
reported as `(2 of 5 runs)` rather than being presented as though it described
all of them.

---

## Reference

### Gas properties

Viscosity, density and compressibility come from
[CoolProp](https://coolprop.org) at the reading's actual temperature and mean
pore pressure — evaluated per reading, not once at startup, since both drift
during a run. Switching the working gas is a config string change. A fixed
viscosity is available as a documented escape hatch, and
`gas.real_gas_correction` divides the reference flow by `Z` when the gas is far
enough from ideal to matter.

### A slow temperature probe

A DS18B20 converts in 750 ms at 12-bit resolution while the DAQ samples every
100 ms, so each temperature is **held** for about eight samples. That is correct,
not a fault: temperature moves far more slowly than the pressures, and viscosity
changes roughly 0.2 % per kelvin.

```yaml
temperature:
  conversion_time_s: 0.75     # DS18B20 at 12-bit; 0.19 s at 9-bit
  warmup_timeout_s: 5.0       # startup wait for the first reading
  stale_after_s: 10.0
  plausible_min_c: -20.0      # excludes the DS18B20 sentinels
  plausible_max_c: 60.0
```

Three things follow from a probe slower than the sample rate:

**The run waits for the first reading** (`Waiting for the temperature probe on
COM4... 0.8 s`). Otherwise the opening fraction of a second would have no
temperature and would silently use `fallback_temperature_c` for the viscosity
lookup — a wrong number, quietly applied.

**A probe that opens but never speaks is fatal** when `temperature.required` is
true, and is caught before the DAQ is touched. A wrong baud rate or a stopped
sketch used to cost a whole run on the fallback.

**Implausible readings are discarded**, keeping the last good value. This matters
specifically for the DS18B20, whose two failure values parse as perfectly
ordinary numbers: `-127` means the sensor did not answer, and `85` is its
power-on reset value. Either would otherwise go straight into the viscosity
lookup. Widen `plausible_min_c` / `plausible_max_c` for a genuinely hot rig.

Every reading records `temperature_age_s`, so the CSV shows the hold directly —
`0.005, 0.106, 0.205 …` resetting each conversion — and the run summary says so
if the probe falls further behind than a few conversions.

### Units

All internal physics runs in **CGS-Darcy** (atm, cm, cP, cm³/s), the units the
Darcy equation was derived in. `gasperm/units.py` is the only module in the
package allowed to hold a conversion constant; everything else converts by
calling through it. Display units are decoupled from both the calibration units
and the internal calculation.
