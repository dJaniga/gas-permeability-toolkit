# gas-permeability-toolkit

Gas permeability measurement for core plugs on a lab rig built around an
**NI USB-6421** DAQ (inlet/outlet pressure, gas flow) and an **Arduino**
temperature probe on USB serial.

Three commands:

| command | what it does |
|---|---|
| `gasperm init` | write the rig and experiment configuration — once per bench |
| `gasperm new-sample` | add a core plug — one file per plug, rig config untouched |
| `gasperm collect` | sample in real time, detect steady state, compute apparent gas permeability with a full uncertainty budget |
| `gasperm klinkenberg` | regress runs at different mean pressures to recover liquid-equivalent permeability `k_L` and slippage factor `b` |

## Install

```bash
pip install -e .          # add [daq] for the nidaqmx driver bindings
pip install -e ".[daq]"
```

`nidaqmx` also needs NI-DAQmx itself, a system-level driver from National
Instruments. It is an optional extra so the physics test suite installs and
runs on a machine that has never seen a DAQ.

## Configuration: three files, three concerns

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

They are split because they change on completely different timescales: the rig
on recalibration, the run on every pressure step, the plug whenever a new one is
loaded.

`gasperm init` writes **only** `hardware.yaml` and `run.yaml` — a sample
describes one plug and a rig measures many, so plugs come from
`gasperm new-sample` instead. `examples/` holds a generated set in exactly this
layout.

Every pressure-bearing field carries its **own** unit, drawn from
`Pa, kPa, MPa, bar, psi, atm`. A rig whose transducers read kPa, an operator who
thinks in bar, and a confining pressure naturally quoted in MPa can coexist
without anyone converting anything by hand.

## Measuring many plugs on one rig

The rig is configured **once**. After that, a new plug is one file and a run is
one command:

```bash
gasperm init tight-gas-rig                       # once per bench; creates the folder
cd tight-gas-rig

gasperm new-sample core-041 --dir samples        # -> samples/core-041.yaml
gasperm new-sample --dir samples --from samples/core-041.yaml   # asks id, then this plug

gasperm collect --sample samples/core-041.yaml --flowmeter low_range
gasperm collect --sample samples/core-041.yaml --flowmeter high_range
gasperm klinkenberg runs/core-041_* --plot
```

`init` prints the exact `new-sample` and `collect` lines for the folder you
named, so the paths are never guesswork.

`--from` carries over what describes the **core** — lithology, formation, well,
depth, grain density, porosity method, who prepared it. It never carries the
id, the geometry, or the per-plug porosity and bulk density: every plug is cut
and measured individually, and inheriting another plug's length or diameter
would put a wrong number straight into the Darcy equation. Those are always
asked for.

`klinkenberg` **refuses** runs from more than one plug unless you pass
`--allow-mixed-samples`. With a directory full of runs from a dozen plugs, a
careless glob is otherwise a silent way to regress across rocks.

### Several flowmeters

A rig usually has more than one meter wired — a low-range and a high-range —
and which one suits a given pressure step is an *experiment* decision, not a
change to the bench. So every meter is defined once in `hardware.yaml`:

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

and each run picks one, in `run.yaml` (`flowmeter: high_range`) or on the
command line (`--flowmeter high_range`). Only the selected meter's analog input
is added to the DAQ task; the others are never read. The meter used is recorded
in every run's metadata, because two runs on the same plug routinely differ in
nothing else.

Each run lands in a timestamped directory containing `readings.csv` (every
sample, raw voltages included, in internal CGS units), `run_metadata.yaml` (a
full config snapshot plus the summary and uncertainty budget) and `run.log`.

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

## Uncertainty (ISO/IEC Guide 98-3, the GUM)

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

`collect` prints the whole budget, ranked by contribution, so it is obvious
which term to improve next.

## Gas properties

Viscosity, density and compressibility come from
[CoolProp](https://coolprop.org) at the reading's actual temperature and mean
pore pressure — evaluated per reading, not once at startup, since both drift
during a run. Switching the working gas is a config string change. A fixed
viscosity is available as a documented escape hatch, and
`gas.real_gas_correction` divides the reference flow by `Z` when the gas is far
enough from ideal to matter.

## Units

All internal physics runs in **CGS-Darcy** (atm, cm, cP, cm³/s), the units the
Darcy equation was derived in. `gasperm/units.py` is the only module in the
package allowed to hold a conversion constant; everything else converts by
calling through it. Display units are decoupled from both the calibration units
and the internal calculation.

## Development

```bash
pytest                    # ~400 tests, no hardware required
ruff check gasperm tests
```

`gasperm/hardware/` is the only package allowed to import `nidaqmx` or
`serial`. Everything else — the physics, the regression, the steady-state
detector, the uncertainty engine — works on plain floats, which is what lets
the whole suite run in CI with nothing plugged in.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
