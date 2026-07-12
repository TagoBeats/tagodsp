# tagodsp

Robins private DSP-Library. Python ist die Workbench, C++ der validierte Core. Kein Code aus NoiseWorks- oder DubCheck-Repos hierher kopieren (IP-Policy, siehe README).

## Kommandos

```bash
uv sync                                          # Deps installieren
uv run pytest                                    # Python-Tests (python/tests/)
cmake -B build cpp && cmake --build build        # C++ Build (header-only, C++20)
ctest --test-dir build                           # C++ Tests (Catch2)
uv run ruff check .                              # Lint (line-length 100)
```

## Struktur

- `python/tagodsp/` — Module nach Domäne: filters/, dynamics/, analysis/, spectral/, synthesis/, utils/. Eine Datei pro Modul.
- `python/tests/` — pytest, Fixtures mit Referenz-Signalen in conftest.py
- `cpp/include/tagodsp/` — header-only Core, nur promotete Module
- `examples/` — ein Render-Script pro Modul (render_listenpack_*.py)
- `listenpacks/` — A/B-Audio + Plots, Ordner `YYYY-MM-DD_HHMM_name/`
- `tools/listenpack.py` — A/B-Renderer
- `docs/concepts/` — Concept Notes mit Quellen, bevor Code entsteht

## Workflow-Regeln

- Python-first: neue Module immer erst in Python, C++ erst nach den Promotion-Kriterien in CONTRIBUTING.md (in echtem Projekt genutzt, API 2+ Wochen stabil, Golden-File-Tests C++ vs Python).
- Keine Bindings: C++ ist eine separate Implementierung, kein pybind11.
- Definition of Done pro Modul: Test mit analytischer Erwartung oder Referenzsignal, Docstring mit Formel + Quelle, Example-Script, bei hörbaren Änderungen ein Listenpack.
- Sample Rate immer explizit als Parameter (`sr`), nie global.
- Naming: Dateien snake_case, Klassen UpperCamelCase. Stateful-Klassen haben `process()` und `reset()`, C++ zusätzlich `prepare(sampleRate, maxBlockSize)`.
- Plots sind Support, Robins Ohr entscheidet. Bei hörbaren Änderungen Listenpack rendern statt aus Plots zu urteilen.
- Scripts (examples/, tools/): CLI-Argumente immer mit argparse, nie manuell aus sys.argv. Pfade mit pathlib.Path, bevorzugt relativ zum Script-File. Keine Emojis im CLI-Output. (Übernommen aus den NoiseWorks Python Script Guidelines, private Nutzung von Moritz freigegeben.)
