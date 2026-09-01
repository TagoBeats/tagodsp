# Hoertests Masking-Detektor, 01.09.2026

Rohdaten zu `docs/concepts/masking_detector.md`. Die Listenpacks selbst liegen unter
`listenpacks/` und sind gitignored (Audio), deshalb stehen die Urteile hier.

Aufbau in beiden Tests: **A** ist das Zielelement allein, **B** dasselbe Element plus die
zweite Spur. Kein RMS-Match, nur ein gemeinsamer Clip-Guard-Gain, damit das Ziel in A und B
bitidentisch ist und ausschliesslich der Maskierer dazukommt. Blind gemischt, feste Seed,
Positionen heissen nur `pos_NN`, Aufloesung erst nach der Abnahme.

---

## Test 1: gemeldete Paare gegen gematchte stille Kontrollen

Pack `2026-08-31_2326_masking_pairs`, Renderer `examples/render_listenpack_masking_pairs.py`.
Drei Arme: nur `relative` meldet, nur `contention` meldet, keins von beiden (Kontrolle).
Jede gemeldete Position mit einer frequenzgematchten Kontrolle (<= 1 Oktave, bevorzugt aus
demselben Beat).

Robin hat pos_02 als nicht wertbar gestrichen, damit faellt Gruppe g3 aus der Auswertung.

| Position | Urteil | Gruppe | Rolle | Arm |
|---|---|---|---|---|
| pos_01 | unsicher | g6 | gemeldet | contention_only |
| pos_02 | unsicher (gestrichen) | g3 | Kontrolle | silent |
| pos_03 | bleibt klar | g2 | Kontrolle | silent |
| pos_04 | bleibt klar | g1 | Kontrolle | silent |
| pos_05 | bleibt klar | g4 | gemeldet | contention_only |
| pos_06 | bleibt klar | g6 | Kontrolle | silent |
| pos_07 | bleibt klar | g2 | gemeldet | relative_only |
| pos_08 | verschwimmt | g3 | gemeldet | relative_only |
| pos_09 | bleibt klar | g5 | Kontrolle | silent |
| pos_10 | verschwimmt | g5 | gemeldet | contention_only |
| pos_11 | verschwimmt | g1 | gemeldet | relative_only |
| pos_12 | verschwimmt | g4 | Kontrolle | silent |

Paarweise:

| Gruppe | Arm | gemeldet | Kontrolle | Befund |
|---|---|---|---|---|
| g1 | relative | verschwimmt | bleibt klar | trennt |
| g2 | relative | bleibt klar | bleibt klar | kein Signal |
| g3 | relative | verschwimmt | gestrichen | nicht auswertbar |
| g4 | contention | bleibt klar | verschwimmt | invertiert |
| g5 | contention | verschwimmt | bleibt klar | trennt |
| g6 | contention | unsicher | bleibt klar | unentschieden |

Kein Pegel-Confound: "verschwimmt" median +6.0 dB breitband (Spanne +1.6 bis +15.9),
"bleibt klar" +8.8 dB (+3.2 bis +14.4), Spannen ueberlappen vollstaendig.

**Das Rangurteil dieses Tests ist entwertet.** Der Kontrollarm war als "meldet keins der
Verfahren" definiert und bestand damit per Konstruktion aus `contention`-Fehlschuessen; das
Design konnte das Verfahren nur schlecht aussehen lassen. Gueltig bleibt die Methodenlehre:
Kontrollen muessen gematcht sein (sonst trennen sich die Arme nach Instrumentengattung), und
das Urteil haengt nicht am Pegelabstand.

---

## Test 2: Matsch gegen Verdeckung, detektorblind

Pack `2026-08-31_2354_mud_vs_burial`, Renderer `examples/render_listenpack_mud_vs_burial.py`.
Auswahl **rein nach dem gemessenen Pegelverhaeltnis im Hauptband des Ziels**, die
Detektor-Scores hatten keinen Einfluss und werden nur nachtraeglich uebergelegt.
Zelle `parity`: `|delta| <= 4 dB`. Zelle `buried`: `12 <= delta <= 20 dB`.

| Position | Urteil | Gruppe | Zelle | delta im Hauptband | contention |
|---|---|---|---|---|---|
| pos_01 | verschwimmt | g6 | Gleichstand | +2.95 dB | 0.930 |
| pos_02 | bleibt klar | g3 | Begraben | +15.12 dB | 0.000 |
| pos_03 | bleibt klar | g2 | Begraben | +13.32 dB | 0.000 |
| pos_04 | bleibt klar | g1 | Begraben | +13.24 dB | 0.000 |
| pos_05 | verschwimmt | g4 | Gleichstand | +0.40 dB | 0.857 |
| pos_06 | bleibt klar | g6 | Begraben | +14.03 dB | 0.000 |
| pos_07 | verschwimmt | g2 | Gleichstand | +1.28 dB | 0.858 |
| pos_08 | unsicher | g3 | Gleichstand | -0.78 dB | 0.000 |
| pos_09 | verschwimmt | g5 | Begraben | +17.27 dB | 0.000 |
| pos_10 | bleibt klar | g5 | Gleichstand | +2.21 dB | 0.000 |
| pos_11 | bleibt klar | g1 | Gleichstand | +0.31 dB | 0.892 |
| pos_12 | verschwimmt | g4 | Begraben | +16.82 dB | 0.000 |

Paarweise, bandgematcht:

| Gruppe | Band | Gleichstand | Begraben | Befund |
|---|---|---|---|---|
| g1 | 56 Hz | bleibt klar (+0.3 dB) | bleibt klar (+13.2 dB) | gleichauf |
| g2 | 56 Hz | verschwimmt (+1.3 dB) | bleibt klar (+13.3 dB) | Gleichstand schlechter |
| g3 | 71 Hz | unsicher (-0.8 dB) | bleibt klar (+15.1 dB) | Gleichstand schlechter |
| g4 | 356 Hz | verschwimmt (+0.4 dB) | verschwimmt (+16.8 dB) | gleichauf |
| g5 | 178 Hz | bleibt klar (+2.2 dB) | verschwimmt (+17.3 dB) | Begraben schlechter |
| g6 | 448 Hz | verschwimmt (+3.0 dB) | bleibt klar (+14.0 dB) | Gleichstand schlechter |

### Befunde

1. **"Geht unter" wurde kein einziges Mal vergeben**, auch nicht in den sechs Positionen, wo
   das Element breitbandig 17 bis 32 dB unter dem Maskierer liegt (im Hauptband 13 bis 17 dB).
   Verdeckung laesst das Element im Spurpaar nicht verschwinden.
2. **Matsch kostet mehr Kontur als Verdeckung:** Gleichstand schlechter in 3 Gruppen, Begraben
   in 1, zweimal gleichauf. Zellsumme Gleichstand 3x verschwimmt von 6, Begraben 2x von 6.
3. **`contention` trennt** (auswertbar, weil die Auswahl detektorblind war): gemeldet 3 von 4
   verschwimmen, still 2 von 8. Auch innerhalb der Gleichstand-Zelle allein trennt es.
4. **Alle Fehlschuesse sind Verdeckungsfaelle.** Jede verschwimmende Position, die `contention`
   nicht meldet, ist ein begrabenes Paar: pos_09 (+17.3 dB) und pos_12 (+16.8 dB) hier,
   pos_12 (+15.9 dB) in Test 1. Drei Faelle, keine Ausnahme.

### Grenzen

Ein Hoerer, zwoelf Positionen pro Test, keine Wiederholungen, keine Statistik. Gerichtete
Evidenz, kein Beweis. Gehoert wird immer nur das Spurpaar, nie der Gesamtmix; das entspricht
dem Produkt (Sidechain-Paar) und nicht der Mischsituation.

Bekannter, nicht entfernbarer Confound in Test 2: in der `buried`-Zelle ist B systematisch
lauter als in der `parity`-Zelle. Das ist dem Phaenomen inhaerent und laesst sich nicht
wegnormieren, ohne das Ziel zu veraendern.
