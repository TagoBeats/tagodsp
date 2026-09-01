# MaskingDetector (analysis/masking.py)

Konflikt-Erkennung zwischen zwei Spuren. Basis fuer X-Ray, Plugin der Tago-Linie
mit Sidechain-Eingang. Portiert aus dem Audiotool-Hackathon-Prototyp
(`project-xray/mockup/analysis/analyze_stems.py`), Clustering 1:1 uebernommen und
gegen den Prototyp verifiziert (26 von 26 Zonen identisch).

## Pipeline

Mono-Summe je Spur, STFT, Leistungsspektrum, Aggregation auf 30 logarithmische
Baender von 20 Hz bis 20 kHz und auf Zeitfenster fester Laenge. Pro Spurpaar
entsteht daraus eine Zellmatrix (Fenster x Band) mit einem Konfliktwert in
[0, 1]. Clustering identisch fuer alle Scorings: BFS ueber Zellen ab `hit_min`,
Bandnachbarschaft 1, Zeitluecken bis 2 Fenster, Mindestdauer 3 Fenster,
Zonen-Score als p90 der Region.

## Drei Scorings, zwei scheiden aus

- `relative` (der ausgelieferte Prototyp-Detektor): jede Spur auf ihren eigenen
  Peak normiert, `cell = min(norm_a, norm_b)`. **Widerlegt.** Im synthetischen
  Pegel-Sweep (`examples/masking_sweep.py`) liefert er ueber 60 dB
  Pegelunterschied hinweg denselben Score, 0.959 konstant von -30 bis +30 dB. Er
  misst "beide haben hier Energie", nicht Verdeckung.
- `collision`: `cell = 2*min(a,b)/(a+b)` plus Hoerbarkeits-Gate. Behebt die
  Pegelblindheit von `relative`, ist als Verhaeltnis aber massstabsblind:
  75 Prozent seiner Zonen liegen in Baendern 25 bis 35 dB unter der
  Breitbandleistung ihres Fensters, also in unhoerbarem Material, das nur
  punktet weil beide Spuren dort gleich leise sind.
- `contention`: `share = min(a,b) / mix_bb[w]`, in dB skaliert auf
  `[contention_floor_db, -3 dB]`. `min(a,b)` ist bereits die streitende Energie,
  `mix_bb` die Breitbandleistung des Paares im Fenster. Braucht kein
  Hoerbarkeits-Gate, weil `min(a,b)` die Arbeit selbst macht: eine Spur viel
  leiser, beide leise, oder eine allein laut ergeben jeweils einen kleinen Wert.
  Die Obergrenze ist exakt -3.01 dB, weil `min(a,b)` hoechstens die Haelfte von
  `mix_bb` sein kann und diese Schranke mit dem Paar als Referenz nicht nur eine
  obere Grenze, sondern erreichbar ist.

## Referenz ist das Paar, nicht der Mix

`mix_bb` ist die Breitbandleistung der zwei verglichenen Spuren, nicht des
Gesamtmix. Grund: das Produkt ist ein Plugin mit Sidechain-Eingang und sieht nie
mehr als zwei Signale. Gemessener Beleg fuer die Notwendigkeit: mit Mix-Referenz
springt ein Paar von 2 auf 11 Zonen, je nachdem wie viele andere Spuren
mitlaufen. Bewusst in Kauf genommener Preis: zwei leise Spuren, die sich
streiten, punkten auch dann, wenn sie unter dem Gesamtmix untergehen.

## Korpus als Negativkontrolle

51 fertige Beats aus `~/Music`, 5076 Spurpaare (`examples/masking_corpus.py`).
`relative` meldet 8.1 Prozent der Paare, `contention` 4.8 Prozent, auf
90.3 Prozent sind beide still. Ueberlappung der gemeldeten Paare 32.3 Prozent,
die Verfahren sind also nicht redundant, sondern verschiedene Detektoren.

## Summen-Stems, die Falle

Der erste Korpuslauf war kontaminiert. FL-Exporte enthalten den Gesamtmix
zweimal (`<beat>_Master` und `<beat>_Current`, bitidentische Dateien, corr 0.967
gegen die Summe der uebrigen Stems bei 44.5 ms Versatz durch PDC). Ein auf
Stringanfang verankertes Namensmuster greift nie, weil FL jedem Stem den
Projektnamen voranstellt. Betroffen: 49 von 51 Beats, 23.7 Prozent aller Paare.
Wirkung auf das Ergebnis: `contention` sah mit 11.5 Prozent lauter aus als
`relative` mit 11.1, sauber gerechnet ist es mit 4.8 gegen 8.1 das stillere
Verfahren. Die Kontamination hatte die Rangfolge vertauscht.

Erkennung deshalb gemessen statt ueber Namen, siehe `analysis/sum_stems.py`:
Kreuzkorrelation gegen die Summe der uebrigen Stems mit Lag-Suche, plus
Partner-Test gegen gedoppelte Layer. Tests in `tests/test_sum_stems.py`.

## Zwei Hoertests, beide blind, 01.09.2026

Aufbau in beiden: A ist das Zielelement allein, B dasselbe Element plus die
zweite Spur. Kein RMS-Match, nur ein gemeinsamer Clip-Guard-Gain, damit das Ziel
in A und B bitidentisch ist und ausschliesslich der Maskierer dazukommt.

- **Test 1** (`render_listenpack_masking_pairs.py`, Pack
  `2026-08-31_2326_masking_pairs`): gemeldete Paare gegen frequenzgematchte
  stille Kontrollen. Sein Rangurteil ueber `contention` ist **entwertet**, weil
  der Kontrollarm als "meldet keins der Verfahren" definiert war und damit per
  Konstruktion aus `contention`-Fehlschuessen bestand. Methodisch gueltig bleibt:
  gematchte Kontrollen sind Pflicht, sonst trennen sich die Arme nach
  Instrumentengattung, und das Urteil haengt nicht am Pegelabstand.
- **Test 2** (`render_listenpack_mud_vs_burial.py`, Pack
  `2026-08-31_2354_mud_vs_burial`): Matsch gegen Verdeckung, Auswahl
  **detektorblind** rein nach dem Pegelverhaeltnis im Hauptband des Ziels.
  Ergebnis: "geht unter" kam in keiner der zwoelf Positionen vor, auch nicht bei
  17 bis 32 dB Abstand breitbandig. Gleichstand kostet oefter Kontur als
  Verdeckung (3 Gruppen gegen 1, zweimal gleichauf). `contention` gemeldet:
  3 von 4 verschwimmen, still: 2 von 8.

Urteile und Auswertung liegen als `verdicts.md` im jeweiligen Listenpack.

## Entscheidung

`contention` ist die Produktbasis. X-Ray meldet **Matsch** (zwei aehnlich laute
Elemente im selben Band), nicht Verdeckung. Die Blindstelle ab etwa +10 dB
Pegelabstand ist dokumentierte Absicht und kostet gemessen rund ein Drittel der
begrabenen Paare: jede verschwimmende Position, die `contention` verpasst hat,
war ein begrabenes Paar (drei Faelle ueber beide Tests, keine Ausnahme).

## Offene Maengel

- Die Skala ist oben unbrauchbar. Nur 2.2 Prozent der Zonen liegen ueber 0.95,
  weil gegen die theoretische Obergrenze -3 dB normiert wird statt gegen einen am
  Korpus gemessenen Wert. Macht eine Anzeigeschwelle wertlos.
- Ein gemeldetes Paar zerfaellt in median 4, maximal 44 Zonen. Das braucht ein
  Zusammenfassen vor der UI, nicht in ihr.

Beides blockiert die Anzeige, nicht den C++-Port.

## Grenzen der Evidenz

Ein Hoerer, zwoelf Positionen pro Test, keine Wiederholungen, keine Statistik.
Gerichtete Evidenz, kein Beweis. Gehoert wird immer nur das Spurpaar, nie der
Gesamtmix; das entspricht dem Produkt und nicht der Mischsituation.

## Status

- Python-Prototyp, offline (`analyze()` auf ganzen Buffern, kein Block-State)
- `hit_min` 0.7 und `score_min` 0.75 stammen aus `relative` und sind fuer
  `contention` unvalidiert
- Skripte: `examples/masking_sweep.py` (Positivkontrolle),
  `examples/masking_corpus.py` (Negativkontrolle in Serie), die beiden
  Listenpack-Renderer fuer die Hoertests
- C++-Promotion erst nach den zwei offenen Maengeln
