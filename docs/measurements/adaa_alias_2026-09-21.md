# ADAA gegen reines Oversampling

Erzeugt von `examples/adaa_alias_sweep.py`, sr 44100 Hz, 216 Messungen.
Metrik ist `alias_nmr_db`: Energie, die weder Grundton noch Harmonische ist, relativ zum Grundton. Kleiner ist sauberer.

## Mittelwert je Konfiguration und Kurve

| Konfiguration | fl | hard | tanh | Mittel |
| :--- | :--- | :--- | :--- | :--- |
| plain 1x | -23.58 | -16.66 | -24.52 | -21.59 |
| plain 4x | -44.79 | -35.86 | -45.67 | -42.11 |
| plain 8x | -51.14 | -40.91 | -52.01 | -48.02 |
| ADAA 1x | -32.71 | -26.31 | -34.17 | -31.06 |
| ADAA 2x | -47.04 | -40.39 | -47.10 | -44.85 |
| ADAA 4x | -53.80 | -45.60 | -53.80 | -51.07 |

## Nach Frequenz, Mittel ueber alle Kurven und Drive-Stufen

| Konfiguration | 1 kHz | 5 kHz | 11 kHz |
| :--- | :--- | :--- | :--- |
| plain 1x | -40.90 | -14.92 | -8.93 |
| plain 4x | -56.23 | -34.88 | -35.21 |
| plain 8x | -56.68 | -36.15 | -51.23 |
| ADAA 1x | -47.71 | -22.50 | -22.99 |
| ADAA 2x | -57.48 | -36.75 | -40.30 |
| ADAA 4x | -56.99 | -36.53 | -59.68 |

## Rechenaufwand je Eingangssample, gezaehlt statt gemessen

| Konfiguration | MACs Resampling | Kennlinien-Auswertungen | Anmerkung |
| :--- | :--- | :--- | :--- |
| plain 4x | 162 | 4 |  |
| plain 8x | 322 | 8 |  |
| ADAA 1x | 0 | 1 | plus 1 Division je Auswertung |
| ADAA 2x | 82 | 2 | plus 1 Division je Auswertung |

Das ist eine Zaehlung, keine Zeitmessung. ADAA ist eine Sample-Rekursion, Oversampling ist vektorisiert, ein numpy-Timing wuerde numpy messen und nicht den Algorithmus. Die CPU-Frage faellt erst in C++.

## Hochton-Abfall, der Preis von ADAA erster Ordnung

Unterhalb des Knies ist ADAA exakt der Zweipunkt-Mittelwert, der Amplitudengang also `cos(pi*f/fs)`. Gemessen und mit der Theorie deckungsgleich:

| Frequenz | ADAA 1x | ADAA 2x | ADAA 4x |
| :--- | :--- | :--- | :--- |
| 5 kHz | -0.56 | -0.14 | -0.03 |
| 10 kHz | -2.42 | -0.56 | -0.14 |
| 15 kHz | -6.35 | -1.30 | -0.31 |
| 20 kHz | -16.74 | -2.42 | -0.56 |

## Treue der 3. Harmonischen gegen plain 8x, 5 kHz bei +11 dB

| Kurve | ADAA 1x | ADAA 2x | ADAA 4x |
| :--- | :--- | :--- | :--- |
| fl | -1.75 | -0.41 | -0.10 |
| hard | -1.62 | -0.38 | -0.09 |
| tanh | -1.72 | -0.40 | -0.10 |
