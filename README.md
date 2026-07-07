# tagodsp

Personal DSP library by Robin Busse. Two tracks:

- **`python/tagodsp`**: the workbench. Fast prototyping, analysis, listening packs. numpy in, numpy out.
- **`cpp/include/tagodsp`**: the core. Header-only C++20, realtime-safe (no allocations, no exceptions in the process path), JUCE-compatible but framework-free. Modules only get promoted here after they are fully validated in Python.

## Why

I build audio tools (de-essing research at NoiseWorks, DubCheck QC app, 7+ years of music production). Every project needs the same building blocks: filters, envelope followers, STFT plumbing, gain math. This library is where they live once, tested and reusable.

## Principles

- Python first. C++ only after a module survived real use.
- Every module ships with tests against analytic expectations or reference signals.
- Audible DSP gets a listening pack (`tools/listenpack.py`). Plots don't have the final word, ears do.
- One module, one file, one job. No framework ambitions.

## Layout

```
python/tagodsp/   utils | filters | analysis | dynamics | spectral | synthesis
python/tests/     pytest, reference signals as fixtures
cpp/include/      header-only C++ core
cpp/tests/        Catch2, golden-file comparison against Python reference
examples/         one script per module, renders listening packs
docs/concepts/    concept notes with literature references, written before code
```

## Setup

```
uv sync
uv run pytest
cmake -B build cpp && cmake --build build && ctest --test-dir build
```
