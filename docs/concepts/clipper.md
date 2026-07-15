# Clipper (distortion/clipper.py)

Statische Clipping-Waveshaper mit optionalem Oversampling. Basis fuer TagoClip
(Plugin Nr. 2 der Tago-Linie).

## Kurvenfamilie

Alle Kurven: ungerade-symmetrisch, Identitaet unterhalb Threshold t, Ceiling 1.0.

- `fl`: exakte Fruity-Soft-Clipper-Kurve, `y = sign(x)(1-(1-t)exp(-(|x|-t)/(1-t)))`
  oberhalb t. Quelle: eigene Messung 15.07.2026 per Testsignal-Bounces
  (TagoClip/measure/analysis/REPORT.md). Beweis: Formel auf kompletten 42-s-Bounce
  angewendet, max. Abweichung 4.7e-8 (Float32-Rauschen). Threshold im Original
  quantisiert auf p/128, Knob-Default p=100.
- `hard`: Hardclip bei +-t
- `tanh`: gleiches Parametrisierungsschema mit tanh-Knie, weicherer Uebergang

## Oversampling

Das Original rechnet ohne Oversampling: Alias der 9. Harmonischen eines
5-kHz-Sinus landet bei 900 Hz mit -32 dB. `Clipper` legt polyphases
Resampling (scipy resample_poly, Kaiser beta 12, ~115 dB Stopband) um die
Nonlinearity. Test: Alias-Reduktion > 40 dB bei os8, 3. Harmonische bleibt
unveraendert (test_clipper.py::test_oversampling_kills_alias).

## Status

- Python-Prototyp, offline (process() auf ganzem Buffer, kein Block-State)
- Clone-Check gegen echte Fruity-Renders: max delta 3-4e-8 auf Sinus- und
  Transient-Material (examples/render_listenpack_clipper.py)
- C++-Promotion: erst nach Nutzung im TagoPitch-artigen Plugin-Kontext,
  dort braucht es Block-Processing mit Resampler-State und Latency-Report
