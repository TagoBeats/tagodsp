# Pitch-Shifting mit Formant-Erhalt (signalsmith-stretch)

Konzept-Note für `pitch/shifter.py` (TagoPitch-Prototyp).

## Problem

Naives Pitch-Shifting (Resampling) verschiebt die Formanten mit: Stimme wird zum
Chipmunk (hoch) oder zu Darth Vader (runter). Ein Vocal-Pitcher à la Little AlterBoy
braucht Pitch und Formant als unabhängige Parameter.

## Ansatz der Library

signalsmith-stretch ist ein Phase-Vocoder-Verwandter ("Signalsmith Stretch"), der
Zeitdehnung und Pitch-Shift im STFT-Bereich macht. Kernideen laut Signalsmith-Doku:

- STFT mit Analyse der Band-zu-Band-Phasenbeziehungen statt klassischem
  Phase-Vocoder-Unwrapping. Reduziert Phasing/Smearing bei polyphonem Material.
- Pitch-Shift über Frequency-Domain-Mapping der Bänder, `tonalityLimit` begrenzt
  das harmonische Mapping nach oben (oberhalb wird verschoben statt skaliert),
  Default-Empfehlung ca. 8 kHz.
- Formant-Kontrolle: Die spektrale Hüllkurve wird geschätzt (Cepstral-artig,
  `setFormantBase` kann die Grundfrequenz als Anker vorgeben) und getrennt vom
  Pitch neu aufgeprägt. `setFormantSemitones(x, compensatePitch)`:
  bei `compensatePitch=true` bleibt die Hüllkurve trotz Transposition stehen
  (Formant-Erhalt), `x` verschiebt sie zusätzlich relativ.

## Bindings (python-stretch, gevendort)

- `third_party/python-stretch`, Upstream-PyPI bindet die Formant-Methoden nicht,
  deshalb gevendort und gepatcht (`setFormantFactor/Semitones/Base` ergänzt).
- `signalsmith-stretch` >= 1.2 hängt von header-only `signalsmith-linear` ab;
  0.3.1 ist unter `include/signalsmith-linear` gevendort, auf Apple mit
  Accelerate-FFT (`SIGNALSMITH_USE_ACCELERATE`).
- `Stretch.process()` ist offline: kompensiert Ein-/Ausgangslatenz intern
  (seek + flush) und resettet den Prozessor nach jedem Aufruf. Ausgabe ist
  zeitlich zum Eingang ausgerichtet, ein Aufruf = ein kompletter Render.
  Für das Plugin (C++) wird später die Streaming-API direkt genutzt.

## Konsequenzen für den Wrapper

- `PitchShifter(sr, ...)` mit `pitch_semitones`, `formant_semitones`, `mix`,
  `preserve_formants` (Default an, das ist der AlterBoy-Modus).
- Latenz (`inputLatency + outputLatency`) abfragbar für spätere Plugin-Reports.
- Mix als linearer Dry/Wet-Crossfade in Python; funktioniert nur weil die
  Bindings die Ausgabe latenzkompensiert liefern.

## Quellen

- https://github.com/Signalsmith-Audio/signalsmith-stretch (README, API-Doku)
- https://signalsmith-audio.co.uk/writing/2023/stretch-design/ (Design-Writeup)
- https://github.com/gregogiudici/python-stretch (Bindings)
- https://github.com/Signalsmith-Audio/linear (FFT/STFT-Backend)
