# gas-permeability-toolkit

Gas permeability of core plugs on a lab rig built around an **NI USB-6421** DAQ
(inlet/outlet pressure, gas flow) and an **Arduino** temperature probe on USB
serial.

Two measurement methods — steady-state Darcy flow, and pulse decay for rock too
tight to measure by flow — with a full ISO/IEC Guide 98-3 uncertainty budget on
every result.

> **This file is the usage guide.** The physics, the metrology and the reasoning
> behind the defaults are in **[MANUAL.md](MANUAL.md)**.

| command | what it does |
|---|---|
| `init` | write the rig and experiment configuration — once per bench |
| `new-sample` | add a core plug — one file per plug, rig config untouched |
| `preview` | watch the raw signals — computes nothing, stores nothing |
| `collect` | acquire, detect steady state, compute permeability with its budget |
| `klinkenberg` | regress runs at different mean pressures for `k_L` and `b` |
| `summarize` | one plug's whole history — every run, the fit, and what is missing |
| `compare` | before/after a treatment, or two plugs, with a paired uncertainty |
| `reprocess` | re-derive stored runs from raw voltages under a changed config |

## Install

```bash
pip install -e .          # add [daq] for the nidaqmx driver bindings
pip install -e ".[daq]"
```

`nidaqmx` also needs NI-DAQmx itself, a system-level driver from National
Instruments. It is an optional extra, so the test suite installs and runs on a
machine that has never seen a DAQ.

## Quick start

The rig is configured **once**. After that a new plug is one file, and a run is
one command.

```bash
gasperm init tight-gas-rig                       # once per bench; creates the folder
cd tight-gas-rig

gasperm preview --plot                           # check the signals; measures nothing
gasperm new-sample core-041 --dir samples        # -> samples/core-041.yaml

# one collect per mean pressure -- at least three for a Klinkenberg fit
gasperm collect --sample samples/core-041.yaml --flowmeter low_range
gasperm collect --sample samples/core-041.yaml --flowmeter low_range
gasperm collect --sample samples/core-041.yaml --flowmeter high_range --stop-after-steady 120

gasperm klinkenberg --sample core-041 --plot     # k_L and b
gasperm summarize core-041                       # everything this plug has been through
```

`init` prints the exact `new-sample` and `collect` lines for the folder you
named, so the paths are never guesswork.

### Rock below ~10 µD

Steady-state flow cannot measure it — the flow is smaller than the flowmeter's
own zero offset, and the symptom is a *negative* `k_L` from runs that every
check passed. Use pulse decay, which measures no flow at all:

```bash
gasperm collect --sample samples/core-041.yaml --leak-test          # do this first
gasperm collect --sample samples/core-041.yaml --method pulse_decay
```

See [Low-permeability rock](MANUAL.md#low-permeability-rock) and
[Pulse decay](MANUAL.md#pulse-decay).

### Before and after a treatment

```bash
gasperm summarize core-041                             # notices two campaigns
gasperm compare core-041 --split 2026-06-01 --plot     # reports the change
```

## Configuration

`gasperm init <folder>` creates a folder per rig, and everything that bench
produces lives inside it:

```
tight-gas-rig/
  hardware.yaml       the bench  -- DAQ, transducer calibrations, every
                                    flowmeter wired, probe, uncertainties
  run.yaml            the run    -- operator, gas, which flowmeter, confining
                                    pressure, steady-state criteria, output
  samples/
    core-041.yaml     the rock   -- one file per core plug: id, lithology,
    core-042.yaml                   geometry, porosity, provenance
  runs/
    core-041_20260803T142530Z/    readings.csv, run_metadata.yaml, run.log
```

Three files because they change on completely different timescales: the rig on
recalibration, the run on every pressure step, the plug whenever a new one is
loaded. `examples/` holds a generated set in this layout.

Each run directory is self-describing: `readings.csv` holds every sample
**including the raw voltages**, and `run_metadata.yaml` holds a full config
snapshot plus the summary and uncertainty budget. That is what lets
`reprocess` re-derive a result under a corrected calibration without repeating
the experiment.

Every pressure-bearing field carries its **own** unit from
`Pa, kPa, MPa, bar, psi, atm`, and plug dimensions likewise
(`mm | cm | m | in | ft`, defaulting to `mm` because that is what a caliper
reads). Nothing has to be converted by hand.

## Commands

Every command takes `-c/--config-dir` to point at a rig folder from anywhere,
and `--help` for the full option list. Only the options you reach for often are
listed here.

### `init`

```bash
gasperm init <folder> [--non-interactive] [--set section.field=value] [--force]
```

Writes **only** `hardware.yaml` and `run.yaml`. Plugs come from `new-sample`,
because a sample describes one plug and a rig measures many. Interactive by
default.

### `new-sample`

```bash
gasperm new-sample core-041 --dir samples
gasperm new-sample --dir samples --from samples/core-041.yaml
```

`--from` carries over what describes the **core** — lithology, formation, well,
depth, grain density, porosity method. It never carries the id, the geometry or
the per-plug porosity: those are always asked for, because inheriting another
plug's length would put a wrong number straight into the Darcy equation.

### `preview`

```bash
gasperm preview                                  # every signal this rig defines
gasperm preview --list                           # what it can show; touches no hardware
gasperm preview -s pulse --plot                  # both pulse-decay transducers
gasperm preview -s inlet_pressure:bar            # one signal, in a unit you choose
gasperm preview --volts                          # raw volts, uncalibrated
gasperm preview -s ai7 -d 30                     # an input the config says nothing about
```

| option | |
|---|---|
| `-s, --signal NAME[:UNIT]` | repeatable; `pulse` and `pressure` select a pair |
| `--list` | the signal catalogue, with each channel and range |
| `--volts` | raw voltage instead of the calibrated value |
| `--plot`, `--plot-window`, `--plot-from-start` | live stacked panels |
| `--rate`, `-d/--duration`, `-n/--samples` | sampling and stop conditions |

Computes nothing and stores nothing — no permeability, no run directory, no
CSV. Only the channels you name are opened, which is what lets you watch the
flowmeter a run is *not* using, or a bare `ai7`, without editing a config file.
Details: [preview](MANUAL.md#preview).

### `collect`

```bash
gasperm collect --sample samples/core-041.yaml
gasperm collect --sample samples/core-041.yaml --method pulse_decay --spacer wide:50
gasperm collect --sample samples/core-041.yaml --leak-test
```

| option | |
|---|---|
| `--sample FILE` | the plug being measured |
| `--method steady_state\|pulse_decay` | overrides `run.yaml` for this run |
| `--flowmeter NAME` | which meter, by name from `hardware.yaml` |
| `--stop-after-steady SECONDS` | end the run once steady state has held that long |
| `-d/--duration`, `-n/--samples` | other stop conditions |
| `--outlet-pressure VALUE` | supply P2 instead of reading the transducer |
| `--leak-test` | the pulse-decay pre-step: measure the *apparatus* |
| `--spacer TYPE:LENGTH` | repeatable; upstream spacers, pulse decay only |
| `--plot`, `--plot-window`, `--plot-from-start`, `--plot-panels` | live view |

The result comes from the **detected steady-state window**; a run that never
settles is written in full but marked not representative, and `klinkenberg`
refuses it unless you pass `--allow-unsteady`. Exit code 2 means the run
produced no confirmed measurement.

Details: [steady state](MANUAL.md#steady-state-is-required-not-optional),
[the live plot](MANUAL.md#watching-it-live),
[a supplied P2](MANUAL.md#a-supplied-downstream-pressure),
[pulse decay](MANUAL.md#pulse-decay).

### `klinkenberg`

```bash
gasperm klinkenberg --sample core-041 --plot
gasperm klinkenberg runs/core-041_2026... runs/core-041_2026...
gasperm klinkenberg --csv points.csv
```

`--sample` finds **every** run for that plug and regresses `k_g` against
`1/P_mean`; the intercept is `k_L` and `slope/intercept` is the slippage factor
`b`. Results go to `runs/klinkenberg_<plug>.yaml` and its `.png`.

It refuses three things unless told otherwise, because each mistake would
otherwise be silent: more than one plug (`--allow-mixed-samples`), mixed P2
conventions (`--allow-mixed-conditions`), and mixed methods
(`--allow-mixed-methods`). Runs that never settled are skipped with a reason;
`--allow-unsteady` includes them. Details:
[Klinkenberg](MANUAL.md#the-klinkenberg-correction).

### `summarize`

```bash
gasperm summarize                       # every plug the runs directory holds
gasperm summarize core-041              # one plug, in full
gasperm summarize core-041 -o core-041.yaml
```

Identity, every run with its result and uncertainty, the fit across them, the
leak tests behind them — and **the gaps**: a run that never confirmed, a series
one pressure short of a fit, two meters where there should be one, a
pulse-decay campaign with no leak test. It also notices when a plug's history is
two campaigns rather than one and points at `compare --split`. Details:
[summarize](MANUAL.md#summarize-and-its-findings).

### `compare`

```bash
gasperm compare core-041 --split 2026-06-01 --plot
gasperm compare core-041 core-042
gasperm compare core-041 --split 2026-06-01 -o change.yaml \
    --label-before as-received --label-after "after 720 h H2"
```

The measurand is the **change**, not either value. Errors common to both
measurements cancel out of their ratio, so a rig reporting `U(k) = 20 %` on each
of two runs can still resolve a 5 % change between them — and every cancellation
is itemised. Exit code 2 means nothing measurable changed. Details:
[comparing two campaigns](MANUAL.md#comparing-two-campaigns).

### `reprocess`

```bash
gasperm reprocess runs/core-041_2026...                        # check it re-derives
gasperm reprocess --sample core-041 --set sample.porosity_uncertainty=0.005
gasperm reprocess --sample core-041 --set sample.length=50.4 --write
gasperm reprocess --sample core-041 --from-config              # after fixing hardware.yaml
```

Re-derives stored runs from their **raw voltages**, so a calibration correction
or a newly measured uncertainty does not mean repeating a fourteen-hour run. It
says which class of change you made — one that moves `k` (a correction), one
that moves only `U(k)` (a re-costing), or neither — and checks that prediction
against the arithmetic.

Reports only, unless `--write`, which writes a **new** `<original>_reprocessed`
directory and never touches the original. Details:
[reprocessing](MANUAL.md#reprocessing-a-stored-run).

## Where the details are

**[MANUAL.md](MANUAL.md)** covers the reasoning, the physics and the failure
modes:

- [Steady state](MANUAL.md#steady-state-is-required-not-optional) — the detector,
  the drift criterion, stopping on a soak
- [Watching it live](MANUAL.md#watching-it-live) — panels, criterion bands, the
  two time views
- [Uncertainty](MANUAL.md#uncertainty-isoiec-guide-98-3) — the GUM budget,
  sensitivity coefficients, correlations
- [Low-permeability rock](MANUAL.md#low-permeability-rock) — why a negative `k_L`
  happens and how to size a meter
- [Pulse decay](MANUAL.md#pulse-decay) — the two models, run times, vessels,
  spacers, transducer sizing, the leak test
- [Comparing two campaigns](MANUAL.md#comparing-two-campaigns) — the paired
  uncertainty and what cancels
- [Reference](MANUAL.md#reference) — gas properties, the temperature probe, units

## Development

```bash
pytest                    # no hardware required
ruff check gasperm tests
```

`gasperm/hardware/` is the only package allowed to import `nidaqmx` or `serial`.
Everything else — the physics, the regression, the steady-state detector, the
uncertainty engine — works on plain floats, which is what lets the whole suite
run in CI with nothing plugged in.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
