# Verteilung der contention-Skala, 01.09.2026

Rohzahlen zu `examples/masking_scale.py`. Versioniert, weil die Ausgabe unter
`listenpacks/` bzw. im Scratchpad liegt und beim naechsten Clone weg waere. Die
Einordnung steht in `docs/concepts/masking_detector.md`, hier stehen nur die
Messwerte.

## Lauf

- Korpus: 51 Beats aus `~/Music`, 5076 Spurpaare, 5 Ordner uebersprungen
- `window_seconds` 0.2, STFT 4096/2048, erste 60 s je Beat
- 30 logarithmische Baender, 20 Hz bis 20 kHz
- Summen-Stems entfernt nach `analysis/sum_stems.py`
- gemessene Groesse: `d = 10*log10(min(a,b) / mix_bb)` in dB, exakte obere
  Schranke -3.01 dB

## Zellen, alle Paare (Perzentile aus einem 0.05-dB-Histogramm)

| Perzentil | d in dB |
|---|---|
| p50 | -72.68 |
| p90 | -35.98 |
| p99 | -15.93 |
| p99.9 | -8.88 |
| p99.99 | -5.78 |
| p100 | -3.13 |

**Hoechste Zelle im ganzen Korpus: -3.13 dB.** Die theoretische Decke ist damit
erreichbares Material und keine reine Rechengroesse.

## Zellen ab der Hit-Schwelle (-11.1 dB)

| Perzentil | d in dB |
|---|---|
| p50 | -9.48 |
| p90 | -6.78 |
| p99 | -4.78 |
| p99.9 | -3.88 |
| p100 | -3.13 |

## Zellen bei Bandgleichstand (r = 2*min/(a+b) >= 0.95)

| Perzentil | d in dB |
|---|---|
| p50 | -44.73 |
| p90 | -15.98 |
| p99 | -7.78 |
| p99.9 | -4.88 |
| p100 | -3.13 |

Zeigt, dass ein Band im Gleichstand allein noch keinen hohen Wert erzeugt: der
Median liegt 40 dB unter der Decke. Was fehlt, ist die Konzentration.

## Bandanteil s = (a+b) / mix_bb

| Perzentil | s in dB |
|---|---|
| p50 | -35.18 |
| p90 | -10.03 |
| p99 | -2.78 |
| p99.9 | -0.63 |
| p100 | -0.03 |

`d` zerfaellt exakt in `r * s / 2`. Die p100-Zeile erklaert, warum die Decke
erreichbar ist: es gibt Fenster, in denen das Paar praktisch seine gesamte
Energie in einem Band hat.

## Gemeldete Zonen (n = 1222)

| | dB | Score alt (Boden -30) | Score neu (Boden -9.75) |
|---|---|---|---|
| min (= Meldeschwelle) | -9.75 | 0.750 | 0.000 |
| median | -7.57 | 0.831 | 0.323 |
| p90 | -5.31 | 0.915 | 0.658 |
| p99 | -4.11 | 0.959 | 0.836 |
| max | -3.10 | 0.996 | 0.985 |

Anteil der Zonen ueber einer Anzeigeschwelle:

| Schwelle | alt | neu |
|---|---|---|
| 0.50 | 100.0 % | 27.7 % |
| 0.75 | 100.0 % | 4.1 % |
| 0.90 | 15.3 % | 0.2 % |
| 0.95 | 2.5 % | 0.1 % |

Keine einzige Zone lag an der Decke an (`clipped_at_ceiling` 0).

## Invarianz-Beleg des Umbaus

Werte-Diff statt gruener Tests, jeweils derselbe Korpus vor und nach der
Aenderung:

- `examples/masking_zone_dump.py`: 1222 Zonen, **0 Abweichungen** in
  `tracks`, `band`, `freq_lo_hz`, `freq_hi_hz`, `w0`, `w1`
- jeder neue Score trifft die vorhergesagte Umrechnung
  `clip((d + 9.75) / 6.75)` auf 2e-6 genau, das ist die Rundung im Dump
- `examples/masking_corpus.py`: 5076 Paare, **0 Abweichungen** in den
  Zonenzahlen, Summary identisch (relative 8.1 %, collision 25.5 %,
  contention 4.8 %, 2 stille Beats)
- veraendert haben sich exakt 242 `top_score`-Werte, alle bei `contention`,
  keiner bei `relative` oder `collision`
