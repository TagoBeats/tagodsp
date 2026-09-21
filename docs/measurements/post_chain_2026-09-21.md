# Traegt die Ceiling-Stufe noch, wenn Plugins folgen?

Material: Summe der echten Stems (Beat `okayes`), 48000 Hz, lauteste Passage. Clipper ADAA 4x, fl-Kurve. Decke -1.0 dBTP. Verifiziert mit 32x.

**Beide Varianten sind nach dem Clipper auf dasselbe RMS gezogen.** Ohne das wuerde die Ceiling-Stufe den Vergleich allein dadurch gewinnen, dass sie leiser ist, und das waere kein Befund, sondern ein Pegelunterschied.

**Pruefung des Messinstruments:** die nachgeschaltete Kette hebt den Crest-Faktor um +3.03 dB. Taete sie das nicht, koennte dieser Test seinen eigenen Gegenstand nicht sehen.

## Alle Laeufe

| Drive | Danach | Ceiling | dBTP vor dem Limiter | Crest | GR max | GR Mittel | GR aktiv % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| +6 | nichts dahinter | aus | +0.85 | +7.84 | +1.90 | +0.59 | 13.3 |
| +6 | nichts dahinter | an | -0.81 | +6.18 | +0.24 | +0.18 | 14.3 |
| +6 | volle Kette | aus | +5.07 | +15.55 | +6.12 | +2.52 | 1.6 |
| +6 | volle Kette | an | +4.09 | +14.61 | +5.14 | +2.18 | 1.5 |
| +12 | nichts dahinter | aus | +1.53 | +5.55 | +2.57 | +0.90 | 47.9 |
| +12 | nichts dahinter | an | -0.16 | +3.86 | +0.89 | +0.77 | 50.1 |
| +12 | volle Kette | aus | +6.28 | +15.70 | +7.33 | +2.17 | 1.9 |
| +12 | volle Kette | an | +5.09 | +14.52 | +6.13 | +2.06 | 2.0 |
| +18 | nichts dahinter | aus | +2.23 | +5.00 | +3.28 | +1.03 | 64.4 |
| +18 | nichts dahinter | an | -0.02 | +2.75 | +1.03 | +0.94 | 66.5 |
| +18 | volle Kette | aus | +6.48 | +15.30 | +7.53 | +2.30 | 3.0 |
| +18 | volle Kette | an | +3.98 | +12.80 | +5.03 | +1.74 | 3.0 |

## Was dem finalen Limiter erspart bleibt

| Drive | Danach | GR max gespart |
| :--- | :--- | :--- |
| +6 | nichts dahinter | +1.66 |
| +6 | volle Kette | +0.98 |
| +12 | nichts dahinter | +1.69 |
| +12 | volle Kette | +1.19 |
| +18 | nichts dahinter | +2.25 |
| +18 | volle Kette | +2.50 |

## Verdikt

**Das Gate ist verfehlt, aber knapp und ungleichmaessig.** Gefordert war 1.0 dB im *schlechtesten* Fall, gemessen sind 0.98 dB. Bei 2 von 3 Drive-Stufen liegt der Wert darueber, Spitze 2.50 dB. Die Schwelle bleibt stehen wie gesetzt: knapp verfehlt ist verfehlt. Aber die Streuung ist selbst der Befund, siehe unten.

Gespart, je Drive-Stufe: +6 dB Drive: 1.66 ohne / 0.98 mit Kette, +12 dB Drive: 1.69 ohne / 1.19 mit Kette, +18 dB Drive: 2.25 ohne / 2.50 mit Kette.

Bei +6 dB, +12 dB frisst die Kette einen Teil des Vorteils, wie erwartet. Bei +18 dB **vergroessert sie ihn**, entgegen der Erwartung: ohne Ceiling-Stufe hinterlaesst der Clipper dort Spitzen, die Transientenformer und Widener anschliessend mit anheben, statt sie zu glaetten.

**Der Vorteil waechst mit dem Drive.** Genau dort, wo hart geclippt wird, traegt die Ceiling-Stufe also am meisten, und zwar durch die Kette hindurch. Wer den Clipper nur antippt, braucht sie nicht.

**Grenze dieser Messung:** der finale Limiter ist die eigene `TruePeakLimiter`-Klasse als Stellvertreter fuer Pro-L 2, und die drei Prozessoren dazwischen sind generische Stellvertreter, keine Modelle der echten Plugins. Die Zahlen zeigen die Richtung und die Groessenordnung, nicht was Pro-L 2 im konkreten Projekt tut.
