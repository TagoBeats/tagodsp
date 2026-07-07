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

## Clean-room rule

Concepts I know from work at NoiseWorks may only enter this library as clean-room reimplementations:

1. Write the concept note in `docs/concepts/` first, from memory and public literature, with citations.
2. Company code stays closed while implementing. No copied names, parameters or defaults.
3. Own API, own test signals, own defaults.
4. De-essing-adjacent modules get an explicit OK from NoiseWorks before the repo goes public.

## Versioning

SemVer. `0.x` until the repo goes public. CHANGELOG from v0.1.
