# Guidelines

These rules keep the library consistent. They apply to me first.

## Python API conventions

- numpy `float32`/`float64` in, numpy out. No implicit resampling, no hidden copies where avoidable.
- Sample rate is always an explicit parameter (`sr`). Never a module-level default.
- No global state. Stateless DSP is a pure function. Stateful DSP is a class with `process()` and `reset()`.
- Docstring includes the formula and the source (RBJ Cookbook, DAFX, paper).

## C++ API conventions

- Header-only, C++20, no dependencies beyond the standard library.
- Lifecycle: `prepare(sampleRate, maxBlockSize)`, `process(...)`, `reset()`.
- No heap allocation, no exceptions, no locks in the process path.

## Naming

- Files and modules: snake_case (`band_ratio_gate.py`, `event_leveler.py`)
- Classes: UpperCamelCase (`BandRatioGate`, `EventLeveler`)
- Tunable parameters live as dataclass fields with validated defaults, never as module-level constants

## Definition of Done (per module)

1. pytest with analytic expectation or reference signal
2. Docstring with formula + source
3. Example script in `examples/`
4. If audible: listening pack rendered and checked by ear

## Python to C++ promotion criteria

Promote only when all of these hold:

1. Used in at least one real project or job
2. API stable for 2+ weeks
3. Python golden files exist; the C++ port must match them within tolerance (test lives in `cpp/tests/`)

## IP rule

NoiseWorks approved (verbal OK Moritz, 2026-07-07; get it in writing before going
public): DSP modules I wrote myself may be ported into this library, and the NW
library conventions may serve as a model.

Hard limits that stay:

1. NW code I did not write myself stays out.
2. DubCheck core algorithms (QC/artifact detection) stay proprietary.
3. Ported modules get adapted to this library's API conventions, not pasted verbatim.

## Versioning

SemVer. `0.x` until the repo goes public. CHANGELOG from v0.1.
